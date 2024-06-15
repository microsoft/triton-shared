//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/OneToNTypeConversion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/PassManager.h"
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

struct ScalarAddptrConverter
    : public OneToNOpConversionPattern<triton::AddPtrOp> {
  using OneToNOpConversionPattern::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }

    auto loc = op->getLoc();

    auto offsetIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), op.getOffset());

    auto layout =
        StridedLayoutAttr::get(op.getContext(), ShapedType::kDynamic, {1});

    auto elemType =
        cast<triton::PointerType>(op.getPtr().getType()).getPointeeType();
    auto memrefType = MemRefType::get({1}, elemType, layout);

    auto ptrInfo = adaptor.getPtr();

    if (ptrInfo.size() == 1) {

      auto castOp = rewriter.create<memref::ReinterpretCastOp>(
          loc, memrefType, ptrInfo[0],
          getAsOpFoldResult(offsetIndex) /*offset*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

      rewriter.replaceOp(op,
                         SmallVector<Value>{castOp.getResult(), offsetIndex},
                         adaptor.getResultMapping());

    } else {
      auto ptr = ptrInfo[0];
      auto offset = ptrInfo[1];

      auto newOffset = rewriter.create<arith::AddIOp>(loc, offset, offsetIndex);

      auto castOp = rewriter.create<memref::ReinterpretCastOp>(
          loc, memrefType, ptr, getAsOpFoldResult(newOffset) /*offset*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

      rewriter.replaceOp(op, SmallVector<Value>{castOp.getResult(), newOffset},
                         adaptor.getResultMapping());
    }

    return success();
  }
};

static std::optional<SmallVector<Value>>
buildGetTupleElementOps(OpBuilder &builder, TypeRange resultTypes, Value input,
                        Location loc) {
  SmallVector<Value> res;
  auto castOp = input.getDefiningOp<UnrealizedConversionCastOp>();

  auto buffer = castOp.getOperand(0);
  auto bufferType = cast<UnrankedMemRefType>(buffer.getType());

  auto layout =
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, {1});

  auto memrefType = MemRefType::get({1}, bufferType.getElementType(), layout);

  auto cast = builder.create<memref::ReinterpretCastOp>(
      loc, memrefType, buffer, 0 /*offset*/, SmallVector<int64_t>{1} /*sizes*/,
      SmallVector<int64_t>{1} /*strides*/);

  res.push_back(cast);
  res.push_back(
      builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0)));
  for (auto t : resultTypes) {
    t.dump();
  }
  return res;
}

static std::optional<Value> buildMakeTupleOp(OpBuilder &builder,
                                             Type resultType, ValueRange inputs,
                                             Location loc) {
  resultType.dump();
  for (auto i : inputs) {
    i.dump();
  }
  return inputs[0];
}

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
    auto moduleOp = getOperation();

    if (failed(convertArgsToMemrefType())) {
      signalPassFailure();
      return;
    }
    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    OneToNTypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [context](triton::PointerType ptrType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          SmallVector<int64_t> strides{1};
          auto layout =
              StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

          auto elemType = ptrType.getPointeeType();
          auto memrefType = MemRefType::get({1}, elemType, layout);

          types = SmallVector<Type>{memrefType, IndexType::get(context)};
          return success();
        });

    converter.addArgumentMaterialization(buildMakeTupleOp);
    converter.addTargetMaterialization(buildGetTupleElementOps);

    patterns.add<ScalarAddptrConverter>(converter, context);

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    // triton::populateStructuredToMemrefConversionPatterns(patterns,
    // converter);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns))))
      return signalPassFailure();

    moduleOp->dump();
    // return;

    // auto moduleOp = getOperation();
    {
      RewritePatternSet patterns(&getContext());
      ConversionTarget target(getContext());

      target.addLegalDialect<
          func::FuncDialect, arith::ArithDialect, math::MathDialect,
          linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
          cf::ControlFlowDialect, tensor::TensorDialect,
          bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
          memref::MemRefDialect>();

      target.addIllegalDialect<tts::TritonStructuredDialect>();

      target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
          [](Operation *op) {
            auto resType = op->getResultTypes()[0];
            return !isa<triton::PointerType>(resType);
          });

      LoopTypeConverter loopTypeConverter(patterns.getContext());

      mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
          loopTypeConverter, patterns, target);

      triton::populateStructuredToMemrefConversionPatterns(patterns,
                                                           loopTypeConverter);

      if (failed(
              applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        signalPassFailure();
      }

      // Erase dead code and fold constants created during lowering
      PassManager pm(&getContext(), moduleOp.getOperationName());
      pm.addPass(createCanonicalizerPass());
      if (failed(runPipeline(pm, getOperation()))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createStructuredToMemrefPass() {
  return std::make_unique<StructuredToMemrefPass>();
}
