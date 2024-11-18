//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtr.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "triton-ptr-to-index"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/Passes.h.inc"

namespace {

static bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });

    addConversion([context](RankedTensorType tensorType)
                      -> std::optional<RankedTensorType> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(tensorType.getShape(),
                                     IntegerType::get(context, 64));
      }
      return std::nullopt;
    });

    addConversion([context](triton::PointerType ptrType) -> Type {
      return IntegerType::get(context, 64);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isPtrTypeLike(type)) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("reconvert-offset-to-ptr",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addTargetMaterialization([&](OpBuilder &builder, IntegerType type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isa<triton::PointerType>(inputs[0].getType())) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("convert-arg-ptr-to-offset",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type type,
                                   ValueRange inputs,
                                   Location loc) -> std::optional<Value> {
      if (inputs.size() == 1 && isPtrTypeLike(type)) {
        auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
        op->setAttr("reconvert-offset-to-ptr",
                    UnitAttr::get(builder.getContext()));
        return op->getResult(0);
      }
      return std::nullopt;
    });
  }
};

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto replacement = rewriter.create<triton::SplatOp>(
        loc, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto replacement = rewriter.create<triton::BroadcastOp>(
        loc, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ptrType = op.getPtr().getType();

    auto cast =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, ptrType, adaptor.getPtr())
            ->getResult(0);

    auto replacement =
        dyn_cast<triton::LoadOp>(rewriter.clone(*op.getOperation()));
    // replacement
    replacement.getPtrMutable().set(cast);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ptrType = op.getPtr().getType();

    auto cast =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, ptrType, adaptor.getPtr())
            ->getResult(0);

    auto replacement =
        dyn_cast<triton::StoreOp>(rewriter.clone(*op.getOperation()));
    // replacement
    replacement.getPtrMutable().set(cast);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  Type getType(Type t) const {
    if (auto shapedType = dyn_cast<ShapedType>(t)) {
      return RankedTensorType::get(shapedType.getShape(),
                                   IntegerType::get(getContext(), 64));
    }
    return IntegerType::get(getContext(), 64);
  }

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ptr = op.getPtr();
    auto func = op->getParentOfType<triton::FuncOp>();
    auto loc = op->getLoc();

    bool isArg = false;
    for (auto arg : func.getArguments()) {
      if (arg == ptr) {
        isArg = true;
        break;
      }
    }

    auto targetType = getType(op.getType());
    Value off = op.getOffset();
    if (targetType != op.getOffset().getType()) {
      off = rewriter.create<arith::ExtSIOp>(loc, targetType, op.getOffset());
    }

    auto prevOff = adaptor.getPtr();
    auto accumulatedOff = rewriter.create<arith::AddIOp>(loc, prevOff, off);
    rewriter.replaceOp(op, accumulatedOff);

    return success();
  }
};

class FoldUnstructuredTritonAddPtrPass
    : public FoldUnstructuredTritonAddPtrBase<
          FoldUnstructuredTritonAddPtrPass> {

  void bfs(Operation *op) {
    std::queue<std::pair<Value, Value>> q;
    DenseSet<Value> visited;

    op->walk([&q, &visited](UnrealizedConversionCastOp op) {
      if (op->hasAttr("convert-arg-ptr-to-offset")) {
        auto value = op->getResult(0);
        q.push({value, op.getInputs()[0]});
        visited.insert(value);
      }
    });

    // // Consider ptrs used directly without addptr
    // op->walk([&q, &visited](triton::FuncOp op) {
    //   for (auto arg : op.getArguments()) {
    //     if (isa<triton::PointerType>(arg.getType())) {
    //       q.push({arg, arg});
    //     }
    //   }
    // });

    while (!q.empty()) {
      auto [v, arg] = q.front();
      // llvm::dbgs() << "visiting: \n";
      // v.dump();
      q.pop();
      for (auto user : v.getUsers()) {
        // scf.for is a special case. We have 2 set of values to consider:
        // - iter-args
        // - loop results
        // for every init arg that originates from a
        // `tts.get_structured_state` op, its corresponding iter-arg and loop
        // result will also be considered "maybeStructured".
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(user)) {
          castOp->dump();
          assert(castOp->hasAttr("reconvert-offset-to-ptr"));
          // assert(castOp.getInputs().size() == 1);
          // assert(castOp->getResults().size() == 1);
          for (auto user : castOp->getUsers()) {
            auto isLoadStore = llvm::isa<triton::LoadOp, triton::StoreOp>(user);
            if (!isLoadStore) {
              user->dump();
              assert(false);
            }
          }

          OpBuilder b(castOp);
          auto createPtrOp = b.create<tts::CreatePtrOp>(
              castOp->getLoc(), castOp->getResult(0).getType(), arg,
              castOp->getOperands()[0]);

          castOp.replaceAllUsesWith(createPtrOp);
        } else if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          auto it = llvm::find(forOp.getInitArgs(), v);
          // assert(0);

          if (it == forOp.getInitArgs().end()) {
            continue;
          }

          // assert(0);
          auto argIndex = std::distance(forOp.getInitArgs().begin(), it);
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);

          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push({neighbor, arg});
            }
          }

        } else {
          for (auto res : user->getResults()) {
            if (res.getType() != v.getType()) {
              // continue;
            }
            if (!visited.contains(res)) {
              visited.insert(res);
              q.push({res, arg});
            }
          }
        }
      }
    }

    op->walk([&q, &visited](UnrealizedConversionCastOp cast) {
      if (cast->hasAttr("convert-arg-ptr-to-offset")) {
        IRRewriter b(cast);
        b.replaceOp(cast, b.create<arith::ConstantOp>(cast->getLoc(),
                                                      b.getI64IntegerAttr(0)));
      }
    });
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  void convertLoop() {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    TritonTypeConverter converter(&getContext());
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect, cf::ControlFlowDialect,
        tensor::TensorDialect, bufferization::BufferizationDialect>();

    target.addLegalOp<ModuleOp>();

    target.addIllegalOp<triton::AddPtrOp>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](Operation *op) {
      auto resType = op->getResultTypes()[0];
      return !isa<triton::PointerType>(resType);
    });

    target.addDynamicallyLegalOp<triton::SplatOp>([](triton::SplatOp op) {
      auto resType = op.getResult().getType();
      if (auto shapedType = dyn_cast<ShapedType>(resType)) {
        return !isa<triton::PointerType>(shapedType.getElementType());
      }
      return !isa<triton::PointerType>(resType);
    });

    target.addDynamicallyLegalOp<triton::BroadcastOp>(
        [](triton::BroadcastOp op) {
          auto resType = op.getResult().getType();
          if (auto shapedType = dyn_cast<ShapedType>(resType)) {
            return !isa<triton::PointerType>(shapedType.getElementType());
          }
          return !isa<triton::PointerType>(resType);
        });

    TritonTypeConverter converter(&getContext());

    patterns.add<AddPtrConverter, SplatConverter, BroadcastConverter>(
        converter, &getContext());
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // not sure why we cannot run this together
    // convertLoop();

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    bfs(moduleOp.getOperation());
    // return;

    moduleOp->dump();

    moduleOp.walk([&](triton::FuncOp func) {
      for (auto arg : func.getArguments()) {
        if (!isa<triton::PointerType>(arg.getType())) {
          continue;
        }

        bool skip = false;
        for (auto user : arg.getUsers()) {
          if (isa<tts::CreatePtrOp>(user)) {
            skip = true;
            break;
          }
        }

        if (skip) {
          continue;
        }

        // ok i remember this now,
        // this is for args that aren't in any of the addptr chain but used in
        // load/store directly
        OpBuilder b(func.getRegion());
        auto loc = func->getLoc();
        auto zero = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
        auto cast = b.create<tts::CreatePtrOp>(loc, arg.getType(), arg, zero);
        arg.replaceAllUsesExcept(cast->getResult(0), cast);
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createFoldUnstructuredTritonAddPtrPass() {
  return std::make_unique<FoldUnstructuredTritonAddPtrPass>();
}
