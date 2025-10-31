//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-arith-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONARITHTOLINALG
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

/// FoldCmpSelectToMinMaxPattern is an optimization pattern that matches
/// 'triton::ReduceOp' operations whose reduction body consists of a
/// single 'arith.select' operation based on a floating-point comparsion,
/// and rewrites them into equivalent 'arith.minf', 'arith.maxf',
/// 'arith.MinimumFOp' or 'arith.MaximumFOp' operations.
///
/// This pattern handles the following cases:
///
/// 1. ** Simple Min/Max Reduction **
///    - select (cmpf ogt a, b), a, b --> maxf(a, b)
///    - select (cmpf olt a, b), a, b --> minf(a, b)
///
/// 2. ** NaN-Aware Min/Max Reduction **
///    - select (cmpf ogt a, b) || cmpf une a, a), a, b --> arith.maximumf(a, b)
///    - select (cmpf olt a, b) || cmpf une a, a), a, b --> arith.minimumf(a, b)
///
/// These transformations not only improve IR canonicalization but also
/// allow the successful lowering of tt.reduce operations to linalg operations,
/// which is already supported in the triton-shared dialect conversion pipeline.

struct FoldCmpSelectToMinMaxPattern
    : public OpRewritePattern<triton::ReduceOp> {
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = *op.getBody();
    auto *term = body.getTerminator();

    // Get the value being yielded from the reduction.
    Value ret = term->getOperand(0);

    // Check if the yielded value is produced by an arith.select operation.
    auto sel = ret.getDefiningOp<arith::SelectOp>();
    if (!sel || !isa<FloatType>(sel.getType()))
      return failure(); // Only handle floating-point types.

    // Extract the condition and operands of the select operation.
    auto cond = sel.getCondition().getDefiningOp();
    Value trueVal = sel.getTrueValue();
    Value falseVal = sel.getFalseValue();

    // Case 1: Simple Min/Max Reduction.
    if (auto cmp = dyn_cast<arith::CmpFOp>(cond)) {
      // Match: select (cmpf ogt a, b), a, b → arith.maxf(a, b).
      if (cmp.getPredicate() == arith::CmpFPredicate::OGT &&
          trueVal == cmp.getLhs() && falseVal == cmp.getRhs()) {
        rewriter.setInsertionPoint(sel);
        auto maxOp =
            rewriter.create<arith::MaxNumFOp>(sel.getLoc(), trueVal, falseVal);
        rewriter.replaceOp(sel, maxOp.getResult());
        return success();
      }
      // Match: select (cmpf olt a, b), a, b → arith.minf(a, b).
      if (cmp.getPredicate() == arith::CmpFPredicate::OLT &&
          trueVal == cmp.getLhs() && falseVal == cmp.getRhs()) {
        rewriter.setInsertionPoint(sel);
        auto minOp =
            rewriter.create<arith::MinNumFOp>(sel.getLoc(), trueVal, falseVal);
        rewriter.replaceOp(sel, minOp.getResult());
        return success();
      }
    }

    // Case 2: NaN-Aware Min/Max Reduction.
    if (auto ori = dyn_cast<arith::OrIOp>(cond)) {
      // Extract both sides of the OR condition.
      auto cmp1 = ori.getLhs().getDefiningOp<arith::CmpFOp>();
      auto cmp2 = ori.getRhs().getDefiningOp<arith::CmpFOp>();
      if (!cmp1 || !cmp2)
        return failure();

      // Helper lambdas to identify comparison patterns.
      auto isOGT = [&](arith::CmpFOp cmp) {
        return cmp.getPredicate() == arith::CmpFPredicate::OGT &&
               trueVal == cmp.getLhs() && falseVal == cmp.getRhs();
      };
      auto isOLT = [&](arith::CmpFOp cmp) {
        return cmp.getPredicate() == arith::CmpFPredicate::OLT &&
               trueVal == cmp.getLhs() && falseVal == cmp.getRhs();
      };
      auto isNaN = [&](arith::CmpFOp cmp) {
        return cmp.getPredicate() == arith::CmpFPredicate::UNE &&
               trueVal == cmp.getLhs() && trueVal == cmp.getRhs();
      };

      // Match: select ((ogt(a, b) || une(a, a)), a, b) -> arith.maximumf(a, b).
      if ((isOGT(cmp1) && isNaN(cmp2)) || (isOGT(cmp2) && isNaN(cmp1))) {
        rewriter.setInsertionPoint(sel);
        auto maxOp =
            rewriter.create<arith::MaximumFOp>(sel.getLoc(), trueVal, falseVal);
        rewriter.replaceOp(sel, maxOp.getResult());
        return success();
      }

      // Match: select ((olt(a, b) || une(a, a)), a, b) -> arith.minimumf(a, b).
      if ((isOLT(cmp1) && isNaN(cmp2)) || (isOLT(cmp2) && isNaN(cmp1))) {
        rewriter.setInsertionPoint(sel);
        auto minOp =
            rewriter.create<arith::MinimumFOp>(sel.getLoc(), trueVal, falseVal);
        rewriter.replaceOp(sel, minOp.getResult());
        return success();
      }
    }
    return failure();
  }
};

class TritonArithToLinalgPass
    : public triton::impl::TritonArithToLinalgBase<TritonArithToLinalgPass> {
  using TritonArithToLinalgBase<
      TritonArithToLinalgPass>::TritonArithToLinalgBase;

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

  // Add additional I32 arguments to represent:
  // - num_programs, 3 in total, one for each axis of the launch grid
  // - program_id, 3 in total, one for each axis of the launch grid
  static void addProgramInfo(triton::FuncOp func) {
    OpBuilder b(func);

    auto origFuncType = func.getFunctionType();
    auto origInputTypes = origFuncType.getInputs();
    SmallVector<Type> newInputTypes(origInputTypes);
    newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

    auto newFuncType =
        b.getFunctionType(newInputTypes, origFuncType.getResults());

    func.setFunctionType(newFuncType);

    // Add empty attributes for each new argument if needed
    if (func.getAllArgAttrs()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      func.getAllArgAttrs(newArgAttrs);
      newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
      func.setAllArgAttrs(newArgAttrs);
    }

    // Add the corresponding arguments to function body
    for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
      func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
    }
  }

  LogicalResult foldCmpSelectToMinMax() {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldCmpSelectToMinMaxPattern>(&getContext());

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

  LogicalResult applyTensorConcatDecomposition() {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return failure();
    }
    return success();
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      populateTritonArithToLinalgCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
        signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        tts::TritonStructuredDialect>();

    target.addLegalOp<ModuleOp>();

    target.addLegalOp<triton::FuncOp, triton::ReturnOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
        [](Operation *op) {
          // Lower dense constant to linalg.fill
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            if (!isa<RankedTensorType>(constOp.getResult().getType())) {
              return true;
            }

            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constOp.getValue())) {
              if (denseAttr.isSplat() &&
                  isa<FloatType, IntegerType>(denseAttr.getElementType())) {
                return false;
              }
            }
            return true;
          }

          bool operateOnTensors =
              llvm::all_of(op->getOperandTypes(), [](Type type) {
                return isa<RankedTensorType>(type);
              });

          return !operateOnTensors;
        });

    if (pidsToFuncArgs) {
      target.addIllegalOp<triton::GetProgramIdOp, triton::GetNumProgramsOp>();
    }

    if (addptrToLinalg) {
      target.addDynamicallyLegalOp<triton::AddPtrOp>([](triton::AddPtrOp op) {
        return !isa<ShapedType>(op.getResult().getType());
      });
    }

    target.addDynamicallyLegalOp<triton::BitcastOp>(
        [this](triton::BitcastOp op) {
          if (!tensorPtrToLinalg) {
            return triton::isPtrTypeLike(op.getType());
          } else {
            if (triton::isPtrTypeLike(op.getType())) {
              return !isa<ShapedType>(op.getType());
            }
            return false;
          }
        });

    // TODO: Might want to consolidate this flag with addptrToLinalg later.
    if (tensorPtrToLinalg) {
      target.addDynamicallyLegalOp<triton::LoadOp, triton::StoreOp,
                                   triton::IntToPtrOp, triton::PtrToIntOp>(
          [](auto op) {
            return !isa<ShapedType>(op->getOperands()[0].getType());
          });
      populateTritonTensorPtrConversionPatterns(patterns);
    }

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    // Fold cmp/select patterns before applying the main conversion patterns.
    if (failed(foldCmpSelectToMinMax())) {
      signalPassFailure();
    }

    triton::populateTritonArithToLinalgConversionPatterns(
        pidsToFuncArgs, addptrToLinalg, assertToCf, transposeReduceToRank0,
        patterns);

    if (pidsToFuncArgs) {
      for (auto func : getOperation().getOps<triton::FuncOp>()) {
        addProgramInfo(func);
      }
    }

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    if (failed(applyTensorConcatDecomposition())) {
      signalPassFailure();
    }

    // Convert tt.func and tt.return into func's counterparts
    if (ttToFuncFunc) {
      moduleOp.walk([&](triton::FuncOp func) {
        OpBuilder builder(func);

        auto name = func.getName();
        auto type = func.getFunctionType();

        SmallVector<DictionaryAttr> argAttrs, resAttrs;
        func.getAllArgAttrs(argAttrs);
        func.getAllResultAttrs(resAttrs);

        auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, type);
        // Preserve the visibility attribute
        funcFunc.setVisibility(func.getVisibility());
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          // Only convert to func.return if the terminator is a tt.return.
          // Otherwise, we will accidentally convert cf.br ops which are also
          // considered terminators.
          if (isa<triton::ReturnOp>(term)) {
            builder.setInsertionPoint(term);
            builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
            term->erase();
          }
        }
        func.erase();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass(bool tensorPtrToLinalg,
                                      bool transposeReduceToRank0) {
  TritonArithToLinalgOptions options;
  options.tensorPtrToLinalg = tensorPtrToLinalg;
  options.transposeReduceToRank0 = transposeReduceToRank0;
  return std::make_unique<TritonArithToLinalgPass>(options);
}
