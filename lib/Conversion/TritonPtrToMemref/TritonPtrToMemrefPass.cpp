//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "triton-ptr-to-index"

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

struct CreatePtrConverter
    : public OpConversionPattern<tts::MakeUnstructuredTensorPtrOp> {
  using OpConversionPattern<
      tts::MakeUnstructuredTensorPtrOp>::OpConversionPattern;

  CreatePtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::MakeUnstructuredTensorPtrOp>(typeConverter,
                                                              context) {}

  CreatePtrConverter(MLIRContext *context)
      : OpConversionPattern<tts::MakeUnstructuredTensorPtrOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::MakeUnstructuredTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct UnrealizedConversionCastOpConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

  UnrealizedConversionCastOpConverter(const TypeConverter &typeConverter,
                                      MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter,
                                                        context) {}

  UnrealizedConversionCastOpConverter(MLIRContext *context)
      : OpConversionPattern<UnrealizedConversionCastOp>(context) {}

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto in = op.getInputs()[0];
    if (auto createPtrOp =
            in.getDefiningOp<tts::MakeUnstructuredTensorPtrOp>()) {
      for (auto user : in.getUsers()) {
        if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(user)) {
          // reinterpretCast.
        }
      }
    }
    return success();
  }
};

static Type getOffsetIndexType(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return RankedTensorType::get(tensorType.getShape(),
                                 IndexType::get(t.getContext()));
  } else if (t.isInteger()) {
    return IndexType::get(t.getContext());
  } else {
    return nullptr;
  }
}

static Value getMemrefArg(Value v) {
  while (auto definingOp = v.getDefiningOp()) {
    assert(isa<UnrealizedConversionCastOp>(definingOp));
    v = definingOp->getOperands()[0];
  }
  return v;
}

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

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    // target.addIllegalOp<UnrealizedConversionCastOp>();

    // patterns.add<UnrealizedConversionCastOpConverter>(patterns.getContext());

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    moduleOp->walk([](tts::MakeUnstructuredTensorPtrOp op) {
      // %4 = "tts.make_unstructured_tptr"(%0, %3) : (!tt.ptr<bf16>, i64) ->
      // !tt.ptr<bf16> %5 = builtin.unrealized_conversion_cast %4 :
      // !tt.ptr<bf16> to memref<*xbf16> %reinterpret_cast =
      // memref.reinterpret_cast %5 to offset: [%c0], sizes: [128, 128],
      // strides: [%c128, %c1] : memref<*xbf16> to memref<128x128xbf16,
      // strided<[?, ?], offset: ?>>
      auto ptr = getMemrefArg(op.getInput());
      auto offset = op.getOffset();
      OpBuilder b(op);

      auto newOffset = b.create<arith::IndexCastOp>(
          op->getLoc(), getOffsetIndexType(offset.getType()), offset);

      SmallVector<Value> workList({op.getPtr()});

      while (!workList.empty()) {
        auto v = workList.back();
        workList.pop_back();
        for (auto user : v.getUsers()) {
          if (auto reinterpretCast =
                  dyn_cast<memref::ReinterpretCastOp>(user)) {
            // reinterpretCast.
            reinterpretCast.getOffsetsMutable()[0].assign(newOffset);
            reinterpretCast.getSourceMutable().assign(ptr);
          } else if (auto unrealizedConversionCast =
                         dyn_cast<UnrealizedConversionCastOp>(user)) {
            assert(unrealizedConversionCast->getResults().size() == 1);
            workList.push_back(unrealizedConversionCast->getResult(0));
          }
        }
      }
    });

    {
      PassManager pm(&getContext(), moduleOp.getOperationName());
      pm.addPass(createCSEPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonPtrToMemrefPass() {
  return std::make_unique<TritonPtrToMemrefPass>();
}
