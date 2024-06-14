//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

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

class TritonFunctionSignatureConverter : public TypeConverter {
public:
  TritonFunctionSignatureConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    // Used for converting memref<*> back to tt.ptr type, these ops will then be
    // handled when we convert addptr op later.
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }
};

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([context](triton::PointerType ptrType) {
      SmallVector<int64_t> strides{1};
      auto layout =
          StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

      auto elemType = ptrType.getPointeeType();
      auto memrefType = MemRefType::get({1}, elemType, layout);
      return memrefType;
    });
  }
};

class TupleTypeConverter : public TypeConverter {
public:
  TupleTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([context](MemRefType memrefType) -> std::optional<TupleType> {
      if (memrefType.hasStaticShape() && memrefType.getRank() == 1) {
        return TupleType::get(context, {memrefType, IndexType::get(context)});
      }
      return std::nullopt;
    });
  }
};

struct LegalizeArithConstantOpsByDecomposition
    : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto base = castOp.getSource();
    const auto offset = castOp.getOffsets()[0];
    SmallVector<Value> replacements{base, offset};
    auto unrealizedOp =
        rewriter
            .create<UnrealizedConversionCastOp>(
                castOp->getLoc(),
                TupleType::get(rewriter.getContext(),
                               {base.getType(), offset.getType()}),
                replacements)
            .getResult(0);
    rewriter.replaceOp(castOp, unrealizedOp);

    // auto vectorType = dyn_cast<VectorType>(constantOp.getType());
    // auto denseAttr =
    // dyn_cast<DenseElementsAttr>(constantOp.getValueAttr()); if
    // (!vectorType || !denseAttr || !denseAttr.isSplat())
    //   return failure();

    // if (!isMultipleOfSMETileVectorType(vectorType))
    //   return rewriter.notifyMatchFailure(constantOp,
    //                                      kMatchFailureNotSMETileTypeMultiple);

    // auto smeTileType =
    // getSMETileTypeForElement(vectorType.getElementType()); auto tileCount
    // = getNumberOfSMETilesForVectorType(vectorType); auto tileSplat =
    // rewriter.create<arith::ConstantOp>(
    //     constantOp.getLoc(), denseAttr.resizeSplat(smeTileType));
    // rewriter.replaceOp(constantOp, SmallVector<Value>(tileCount,
    // tileSplat),
    //                    adaptor.getResultMapping());

    return success();
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

  void runOnOperation() override {

    if (failed(convertArgsToMemrefType())) {
      signalPassFailure();
      return;
    }

    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();

    target.addIllegalDialect<tts::TritonStructuredDialect>();

    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([](Operation *op) {
      auto resType = op->getResultTypes()[0];
      return !isa<triton::PointerType>(resType);
    });

    LoopTypeConverter loopTypeConverter(patterns.getContext());
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        loopTypeConverter, patterns, target);
    triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                         loopTypeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Erase dead code and fold constants created during lowering
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    {
      MLIRContext *context = &getContext();
      OneToNTypeConverter converter;
      RewritePatternSet patterns(&getContext());
      converter.addConversion([](Type type) { return type; });
      converter.addConversion([context](MemRefType memrefType,
                                        SmallVectorImpl<Type> &types)
                                  -> std::optional<LogicalResult> {
        if (memrefType.hasStaticShape() && memrefType.getRank() == 1) {
          types = SmallVector<Type>{
              TupleType::get(context, {memrefType, IndexType::get(context)})};
          return success();
        }

        return std::nullopt;
      });

      // converter.addSourceMaterialization(
      //     [&](OpBuilder &builder, Type resultType, ValueRange inputs,
      //         Location loc) -> std::optional<Value> {
      //       return builder
      //           .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      //           .getResult(0);
      //     });

      // converter.addArgumentMaterialization(
      //     [&](OpBuilder &builder, TupleType resultType, ValueRange inputs,
      //         Location loc) {
      //       return builder
      //           .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      //           .getResult(0);
      //     });

      // converter.addTargetMaterialization(
      //     [&](OpBuilder &builder, TypeRange resultTypes, Value input,
      //         Location loc) -> std::optional<SmallVector<Value>> {
      //       return SmallVector<Value>{
      //           builder
      //               .create<UnrealizedConversionCastOp>(loc, resultTypes,
      //               input) .getResult(0)};
      //     });
      // ConversionTarget target(getContext());

      scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
      // patterns.add<LegalizeArithConstantOpsByDecomposition>(converter,
      // context);
      if (failed(applyPartialOneToNConversion(moduleOp, converter,
                                              std::move(patterns)))) {
        signalPassFailure();
      }
    }

    // {
    //   MLIRContext *context = &getContext();
    //   OneToNTypeConverter converter;
    //   RewritePatternSet patterns(&getContext());
    //   converter.addConversion([](Type type) { return type; });
    //   converter.addConversion(
    //       [context](MemRefType memrefType, SmallVectorImpl<Type> &types)
    //           -> std::optional<LogicalResult> {
    //         if (memrefType.hasStaticShape() && memrefType.getRank() == 1) {
    //           types = SmallVector<Type>{memrefType, IndexType::get(context)};
    //           return success();
    //         }
    //         return std::nullopt;
    //       });

    //   scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    //   patterns.add<LegalizeArithConstantOpsByDecomposition>(converter,
    //   context); if (failed(applyPartialOneToNConversion(getOperation(),
    //   converter,
    //                                           std::move(patterns))))
    //     return signalPassFailure();
    // }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
