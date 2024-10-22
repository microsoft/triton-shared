//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToIndex/TritonPtrToIndex.h"
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
#include "triton-shared/Conversion/TritonPtrToIndex/Passes.h.inc"

namespace {

// Type getIndexType() {
//   auto pointeeType = ptrType.getPointeeType();
//   if (auto shapedType = dyn_cast<ShapedType>(pointeeType)) {
//     return RankedTensorType::get(shapedType.getShape(),
//                                  IndexType::get(context));
//   }
//   return IndexType::get(context);
// }

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](RankedTensorType tensorType)
                      -> std::optional<RankedTensorType> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return RankedTensorType::get(tensorType.getShape(),
                                     IntegerType::get(context, 32));
      }
      return std::nullopt;
    });

    addConversion([context](triton::PointerType ptrType) -> Type {
      return IntegerType::get(context, 32);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          ->getResult(0);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type type,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          ->getResult(0);
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

struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  Type getType(Type t) const {
    if (auto shapedType = dyn_cast<ShapedType>(t)) {
      return RankedTensorType::get(shapedType.getShape(),
                                   IntegerType::get(getContext(), 32));
    }
    return IntegerType::get(getContext(), 32);
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
    // if (targetType != op.getType()) {
    //   off =
    //       rewriter.create<arith::IndexCastOp>(loc, targetType,
    //       op.getOffset());
    // }
    if (isArg) {
      // starts from 0
      rewriter.replaceOp(op, off);
      // rewriter.eraseOp(op);
    } else {
      auto prevOff = adaptor.getPtr();
      auto accumulatedOff = rewriter.create<arith::AddIOp>(loc, prevOff, off);
      rewriter.replaceOp(op, accumulatedOff);
      // rewriter.eraseOp(op);
    }

    return success();
  }
};

class TritonPtrToIndexPass : public TritonPtrToIndexBase<TritonPtrToIndexPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
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

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToIndexPass() {
  return std::make_unique<TritonPtrToIndexPass>();
}
