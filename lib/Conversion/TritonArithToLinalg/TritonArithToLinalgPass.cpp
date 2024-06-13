//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
      if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
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

    if (!assertToCf) {
      target.addLegalOp<triton::AssertOp>();
    }

    triton::populateTritonArithToLinalgConversionPatterns(
        pidsToFuncArgs, addptrToLinalg, assertToCf, patterns);

    if (pidsToFuncArgs) {
      for (auto func : getOperation().getOps<triton::FuncOp>()) {
        addProgramInfo(func);
      }
    }

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
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
        funcFunc.setAllArgAttrs(argAttrs);
        funcFunc.setAllResultAttrs(resAttrs);

        auto &funcFuncBody = funcFunc.getBody();
        auto &funcBody = func.getBody();

        IRMapping map;
        funcBody.cloneInto(&funcFuncBody, map);

        for (Block &block : funcFuncBody.getBlocks()) {
          auto term = block.getTerminator();
          builder.setInsertionPoint(term);
          builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
          term->erase();
        }
        func.erase();
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonArithToLinalgPass() {
  return std::make_unique<TritonArithToLinalgPass>();
}
