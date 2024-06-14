//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/UseAnalysis.h"
#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalg/Passes.h.inc"

namespace {

struct Reinterpret : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto base = castOp.getSource();
    const auto offset = castOp.getOffsets()[0];
    SmallVector<Value> replacements{base, offset};
    auto unrealizedOp = rewriter
                            .create<UnrealizedConversionCastOp>(
                                castOp->getLoc(),
                                TupleType::get(rewriter.getContext(),
                                               {castOp.getResult().getType(),
                                                offset.getType()}),
                                replacements)
                            .getResult(0);

    rewriter.replaceAllUsesWith(castOp, unrealizedOp);
    rewriter.eraseOp(castOp);
    // rewriter.replaceOp(castOp, unrealizedOp);

    return success();
  }
};

struct ExtractStrided
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extract, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto src = adaptor.getSource();
    src.dump();
    auto base = extract.getBaseBuffer();
    auto offset = extract.getOffset();

    auto prevOffset =
        rewriter
            .create<UnrealizedConversionCastOp>(
                extract->getLoc(), offset.getType(), SmallVector<Value>{src})
            .getResult(0);

    auto prevBuffer =
        rewriter
            .create<UnrealizedConversionCastOp>(
                extract->getLoc(), base.getType(), SmallVector<Value>{src})
            .getResult(0);

    // rewriter.replaceOp(extract, prevOffset);
    rewriter.replaceAllUsesWith(base, prevBuffer);
    rewriter.replaceAllUsesWith(offset, prevOffset);
    rewriter.eraseOp(extract);

    return success();
  }
};

class LoopTypeConverter : public TypeConverter {
public:
  LoopTypeConverter(MLIRContext *context) {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    // addConversion([context](TupleType type) {
    //   return TupleType::get(context, {memrefType, IndexType::get(context)});
    // });
    addConversion([context](MemRefType memrefType) {
      return TupleType::get(context, {memrefType, IndexType::get(context)});
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // addTargetMaterialization([&](OpBuilder &builder, Type resultType,
    //                              ValueRange inputs,
    //                              Location loc) -> std::optional<Value> {
    //   return builder.create<UnrealizedConversionCastOp>(loc, resultType,
    //   inputs)
    //       .getResult(0);
    // });

    // addArgumentMaterialization([&](OpBuilder &builder, Type resultType,
    //                                ValueRange inputs,
    //                                Location loc) -> std::optional<Value> {
    //   return builder.create<UnrealizedConversionCastOp>(loc, resultType,
    //   inputs)
    //       .getResult(0);
    // });
  }
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(&getContext());
      ConversionTarget target(getContext());
      target.addIllegalOp<memref::ReinterpretCastOp>();
      // target.addIllegalOp<memref::ExtractStridedMetadataOp>();
      target.addLegalOp<UnrealizedConversionCastOp>();
      LoopTypeConverter loopTypeConverter(patterns.getContext());
      patterns.add<Reinterpret>(context);

      mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
          loopTypeConverter, patterns, target);
      if (failed(
              applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        signalPassFailure();
      }
      return;
    }

    OneToNTypeConverter converter;
    RewritePatternSet patterns(&getContext());
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [context](MemRefType memrefType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          types = SmallVector<Type>{
              TupleType::get(context, {memrefType, IndexType::get(context)})};
          return success();
        });

    converter.addSourceMaterialization(
        [&](OpBuilder &builder, Type resultType, ValueRange inputs,
            Location loc) -> std::optional<Value> {
          return builder
              .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
              .getResult(0);
        });

    converter.addArgumentMaterialization([&](OpBuilder &builder,
                                             TupleType resultType,
                                             ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    converter.addTargetMaterialization(
        [&](OpBuilder &builder, TypeRange resultTypes, Value input,
            Location loc) -> std::optional<SmallVector<Value>> {
          return SmallVector<Value>{
              builder
                  .create<UnrealizedConversionCastOp>(loc, resultTypes, input)
                  .getResult(0)};
        });
    // ConversionTarget target(getContext());
    // patterns.add<Reinterpret, ExtractStrided>(context);

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    if (failed(applyPartialOneToNConversion(moduleOp, converter,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
