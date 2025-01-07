//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <optional>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_STRUCTUREDTOMEMREF
#include "triton-shared/Conversion/StructuredToMemref/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    // addConversion([context](triton::PointerType ptrType) {
    //   SmallVector<int64_t> strides{1};
    //   auto layout =
    //       StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

    //   auto elemType = ptrType.getPointeeType();
    //   auto memrefType = MemRefType::get({1}, elemType, layout);
    //   return memrefType;
    // });

    // A tensor of pointers can be passed in as scf.for's init-args, in such
    // cases, we convert the type to a memref with dynamic offsets and
    // strides.
    addConversion(
        [context](RankedTensorType tensorType) -> std::optional<MemRefType> {
          if (auto ptrType = llvm::dyn_cast<triton::PointerType>(
                  tensorType.getElementType())) {
            auto layout = StridedLayoutAttr::get(
                context, ShapedType::kDynamic,
                SmallVector<int64_t>(tensorType.getRank(),
                                     ShapedType::kDynamic));
            Type elemType = ptrType.getPointeeType();
            return MemRefType::get(tensorType.getShape(), elemType, layout);
          }

          return std::nullopt;
        });

    // Convert the current memref type to a memref type with dynamic offsets and
    // strides through another reinterpret_cast with the same offsets.
    // Canonicalization will simplify this sequence by removing the inital
    // reinterpret_cast.
    addTargetMaterialization([&](OpBuilder &builder, MemRefType memrefType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      auto reinterpretCast =
          inputs[0].getDefiningOp<memref::ReinterpretCastOp>();
      if (!reinterpretCast) {
        return builder
            .create<UnrealizedConversionCastOp>(loc, memrefType, inputs)
            .getResult(0);
      }
      return builder.create<memref::ReinterpretCastOp>(
          loc, memrefType, inputs[0], reinterpretCast.getMixedOffsets()[0],
          reinterpretCast.getMixedSizes(), reinterpretCast.getMixedStrides());
    });

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

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class StructuredToMemrefPass
    : public triton::impl::StructuredToMemrefBase<StructuredToMemrefPass> {
  using StructuredToMemrefBase<StructuredToMemrefPass>::StructuredToMemrefBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, triton::TritonDialect,
                    ttx::TritonTilingExtDialect, memref::MemRefDialect>();
  }

  LogicalResult convertArgsToMemrefType() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TritonFunctionSignatureConverter typeConverter;

    // Update function signature to use memrefs
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);

    return applyPartialConversion(moduleOp, target, std::move(patterns));
  }

  // We leverage the 1->N conversion infrastructure to convert tt.addptr for
  // scalar to memref.reinterpret_cast.
  //
  // A tt.addptr has the following form:
  //
  // %new_ptr = tt.addptr %ptr %offset
  //
  // where %new_ptr and %ptr have tt.ptr type, and %offset is of index type.
  //
  // With this form, there can be a chain of tt.addptr where we keep adding
  // offsets to an existing pointer:
  //
  // %ptr_1 = tt.addptr %arg0 %offset
  // %ptr_2 = tt.addptr %ptr_1 %offset
  // %ptr_3 = tt.addptr %ptr_2 %offset
  //
  // Now, we want to lower each tt.addptr to a memref.reinterpret_cast so that
  // the pointers can be used by affine.load and affine.store (lowered from
  // tt.load and tt.store).
  //
  // A memref.reinterpret_cast op also takes an offset and returns a memref in a
  // similar fashion to tt.addptr:
  //
  //  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes:
  //  [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset:
  //  ?>>
  //
  // However, since the semantic of memref.reinterpret_cast is different,
  // the following lowering would be incorrect for the sequence of tt.addptr
  // above:
  //
  //  %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_2 = memref.reinterpret_cast %cast_1 to offset [%offset]
  //  %cast_3 = memref.reinterpret_cast %cast_2 to offset [%offset]
  //
  // The above sequence is equivalent to:
  //
  //  %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_2 = memref.reinterpret_cast %arg0 to offset [%offset]
  //  %cast_3 = memref.reinterpret_cast %arg0 to offset [%offset]
  //
  // In other word, memref.reinterpret_cast ignores the current offset of the
  // input buffer.
  //
  // Therefore, we have to manually track the offset for each addptr by lowering
  // to the following form:
  //
  // %offset_1 = arith.addi %cst_0 %offset
  // %cast_1 = memref.reinterpret_cast %arg0 to offset [%offset_1]
  //
  // %offset_2 = arith.addi %offset_1 %offset
  // %cast_2 = memref.reinterpret_cast %arg0 to offset [%offset_2]
  //
  // %offset_3 = arith.addi %offset_2 %offset
  // %cast_3 = memref.reinterpret_cast %arg0 to offset [%offset_3]
  //
  // Each tt.addptr is lowered to a pair of arith.addi that accumulates the
  // current offset before using that offset to the reinterpret_cast.
  LogicalResult convertAddPtrToReinterpretCast() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->2 type conversion here, where a triton pointer type
    // maps to a pair of {memref, index} type for the the buffer and offset.
    converter.addConversion(
        [context](triton::PointerType ptrType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          types = SmallVector<Type>{getMemrefTypeForScalarPtr(ptrType, context),
                                    IndexType::get(context)};
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a pair of {memref,
    // index} type back to the original triton pointer type.
    // These are used when there are ops that still need to use the original
    // pointer type. For instance, we convert the result of tt.addptr from
    // tt.ptr type to a pair of {memref, index}, but the original ptr result is
    // still being used by another tt.load or tt.store.
    converter.addArgumentMaterialization(buildCastOp);
    converter.addSourceMaterialization(buildCastOp);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a pair of {memref, index} type.
    converter.addTargetMaterialization(buildCastAndOffsetOps);

    patterns.add<ScalarAddptrConverter>(converter, context);

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      return failure();
    }

    return success();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalOp<tts::LoadOp, tts::StoreOp, tts::MakeTensorPtrOp>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                         typeConverter);

    LoopTypeConverter loopTypeConverter(patterns.getContext());

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        loopTypeConverter, patterns, target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
