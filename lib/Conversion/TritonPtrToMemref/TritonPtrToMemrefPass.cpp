//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

namespace {

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(), ptrType.getPointeeType());
      }
      return std::nullopt;
    });
    // Used for converting memref<*> back to tt.ptr type, these ops will then be
    // handled when we convert addptr op later.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
                                   ValueRange inputs,
                                   Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

struct FromElementsConverter
    : public OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

  FromElementsConverter(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<tensor::FromElementsOp>(typeConverter, context) {}

  FromElementsConverter(MLIRContext *context)
      : OpConversionPattern<tensor::FromElementsOp>(context) {}

  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(),
        MemRefType::get(op.getType().getShape(),
                        UnrankedMemRefType::get(rewriter.getI32Type(), 0)));

    alloc->dump();
    for (auto i = 0; i < adaptor.getElements().size(); i++) {
      Value index = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      auto elem = adaptor.getElements()[i];
      rewriter.create<memref::StoreOp>(op.getLoc(), elem, alloc,
                                       ValueRange{index});
    }
    rewriter.replaceOp(op, alloc);
    return llvm::success();
  }
};

struct ExtractOpConverter : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

  ExtractOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tensor::ExtractOp>(typeConverter, context) {}

  ExtractOpConverter(MLIRContext *context)
      : OpConversionPattern<tensor::ExtractOp>(context) {}

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getTensor();
    auto ptrType =
        dyn_cast<triton::PointerType>(srcType.getType().getElementType());

    if (!ptrType) {
      return failure();
    }

    auto extract = rewriter.create<memref::LoadOp>(
        op.getLoc(), adaptor.getTensor(), adaptor.getIndices());

    rewriter.replaceOp(op, extract);

    return llvm::success();
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  UnrealizedCastConverter(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter,
                                                        context) {}

  UnrealizedCastConverter(MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(context) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());

    return llvm::success();
  }
};

class TritonPtrToMemrefPass
    : public TritonPtrToMemrefBase<TritonPtrToMemrefPass> {

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
    TritonFunctionSignatureConverter typeConverter;

    patterns.add<FromElementsConverter, ExtractOpConverter,
                 UnrealizedCastConverter>(patterns.getContext());

    target.addDynamicallyLegalOp<tensor::ExtractOp>([](tensor::ExtractOp op) {
      return !isa<triton::PointerType>(
          op.getTensor().getType().getElementType());
    });

    target.addIllegalOp<tensor::FromElementsOp>();
    target.addLegalOp<memref::StoreOp, memref::AllocOp, arith::ConstantIndexOp,
                      memref::LoadOp>();
    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // PassManager pm(&getContext(), moduleOp.getOperationName());
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(createCSEPass());
    // if (failed(runPipeline(pm, getOperation()))) {
    //   signalPassFailure();
    // }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToMemrefPass() {
  return std::make_unique<TritonPtrToMemrefPass>();
}
