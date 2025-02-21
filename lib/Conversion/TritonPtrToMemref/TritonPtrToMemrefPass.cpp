//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>

#define DEBUG_TYPE "triton-ptr-to-memref"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h.inc"

namespace {

struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  AddPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::AddPtrOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto pointeeType = cast<triton::PointerType>(op.getType()).getPointeeType();
    auto offsetType = op.getOffset().getType();
    auto pointeeSizeInBytes = rewriter.create<arith::ConstantIntOp>(
        loc, pointeeType.getIntOrFloatBitWidth() / 8, offsetType);
    auto scaledOffset =
        rewriter.create<arith::MulIOp>(loc, op.getOffset(), pointeeSizeInBytes);
    auto add = rewriter.create<tptr::PtrAddOp>(
        loc, ptr::PtrType::get(rewriter.getContext()), adaptor.getPtr(),
        scaledOffset);
    rewriter.replaceOp(op, add);
    return success();
  }
};

struct BitCastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  BitCastConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::BitcastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrLoad = rewriter.create<tptr::LoadOp>(op.getLoc(), op.getType(),
                                                 adaptor.getPtr());
    rewriter.replaceOp(op, ptrLoad);
    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  StoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeOp = rewriter.create<tptr::StoreOp>(op->getLoc(), op.getValue(),
                                                  adaptor.getPtr());
    rewriter.replaceOp(op, storeOp);
    return success();
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  UnrealizedCastConverter(const TypeConverter &typeConverter,
                          MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter,
                                                        context) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getInputs().front();
    auto output = op.getResult(0);
    if (isa<UnrankedMemRefType>(input.getType()) &&
        isa<triton::PointerType>(output.getType())) {
      rewriter.replaceOp(op, adaptor.getInputs().front());
      return success();
    } else if (isa<triton::PointerType>(input.getType()) &&
               isa<UnrankedMemRefType>(output.getType())) {
      op->dump();
    }

    // assert(0);
    return failure();
  }
};

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(),
                                     /*memorySpace=*/0);
    });
    addConversion([](RankedTensorType tensorType) -> std::optional<Type> {
      if (auto ptrType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(), ptrType.getPointeeType());
      }
      return std::nullopt;
    });

    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
    addArgumentMaterialization(createUnrealizedCast);
    // addTargetMaterialization([&](OpBuilder &builder, Type resultType,
    //                              ValueRange inputs, Location loc) -> Value {
    //   return builder.create<tptr::FromMemrefOp>(loc, resultType, inputs)
    //       .getResult();
    // });
  }
};

class TritonPtrTypeConverter : public TypeConverter {
public:
  TritonPtrTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      return ptr::PtrType::get(context);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<tptr::FromMemrefOp>(loc, resultType, inputs)
          .getResult();
    });
    auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                    ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
    addArgumentMaterialization(createUnrealizedCast);
  }
};

class TritonPtrToMemrefPass
    : public TritonPtrToMemrefBase<TritonPtrToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect, tptr::TPtrDialect>();
  }

  void convertTritonPtrToMemref() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;

    // Update function signature and call ops to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp, triton::FuncOp>([&](auto op) {
      return typeConverter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op.getResultTypes()) &&
             typeConverter.isLegal(op.getOperandTypes());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }

  void convertTritonPtr() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonPtrTypeConverter typeConverter(&getContext());

    // TODO: This needs to be a dynamic check, we want to lower any ops with
    // pointer operands
    target.addIllegalOp<triton::AddPtrOp, triton::BitcastOp, triton::LoadOp,
                        triton::StoreOp>();
    target.addLegalDialect<tptr::TPtrDialect>();
    // target.addLegalOp<UnrealizedConversionCastOp>();
    // target.addDynamicallyLegalOp<tptr::FromMemrefOp>(
    //     [&](tptr::FromMemrefOp op) { return op.getType(); });

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          auto input = op.getInputs().front();
          auto output = op.getResult(0);
          if (isa<UnrankedMemRefType>(input.getType()) &&
              isa<triton::PointerType>(output.getType())) {
            return false;
          }
          return true;
        });
    target.addLegalDialect<arith::ArithDialect>();

    patterns.add<AddPtrConverter, BitCastConverter, StoreConverter,
                 LoadConverter, UnrealizedCastConverter>(typeConverter,
                                                         patterns.getContext());

    // patterns.add<UnrealizedCastConverter>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    convertTritonPtrToMemref();
    convertTritonPtr();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToMemrefPass() {
  return std::make_unique<TritonPtrToMemrefPass>();
}
