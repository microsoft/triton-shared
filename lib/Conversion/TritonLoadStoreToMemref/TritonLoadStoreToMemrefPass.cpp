//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton-shared/Analysis/UseAnalysis.h"
#include "triton-shared/Conversion/TritonLoadStoreToMemref/TritonLoadStoreToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonLoadStoreToMemref/Passes.h.inc"

namespace {

class EmptyConverter : public TypeConverter {
public:
  EmptyConverter() {
    // // The order of type conversion is important: later ones are tried
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
    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // addSourceMaterialization([&](OpBuilder &builder, Type resultType,
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

static MemRefType getMemrefTypeForScalarPtr(triton::PointerType ptrType,
                                            MLIRContext *context) {
  SmallVector<int64_t> strides{1};
  auto layout = StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);

  auto elemType = ptrType.getPointeeType();
  auto memrefType = MemRefType::get({1}, elemType, layout);
  return memrefType;
}

struct ScalarLoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  ScalarLoadConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  ScalarLoadConverter(MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto castOp = op.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto results = op->getResultTypes();

    auto loc = op->getLoc();
    auto basePtr = castOp.getInput();
    auto offsets = castOp.getOffset();

    Value loadIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offsets);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(op.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(loadIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    auto loadOp = rewriter.create<affine::AffineLoadOp>(loc, memref, zeroMap,
                                                        std::nullopt);
    rewriter.replaceOp(op, loadOp.getResult());

    return success();
  }
};

struct ScalarStoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  ScalarStoreConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context) {}

  ScalarStoreConverter(MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto castOp = op.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto results = op->getResultTypes();

    auto loc = op->getLoc();
    auto basePtr = castOp.getInput();
    auto offsets = castOp.getOffset();

    Value storeIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offsets);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(op.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(storeIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto val = op.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    rewriter.create<affine::AffineStoreOp>(loc, val, memref, zeroMap,
                                           std::nullopt);
    rewriter.eraseOp(op);

    return success();
  }
};

struct LoadOpConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LoadOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LoadOpConverter(MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto op = loadOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto results = op->getResultTypes();

    auto loc = op->getLoc();
    auto ptr = op.getInput();
    auto offsets = op.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsets.getType());

    if (!offsetType) {
      return failure();
    }

    // auto ptrType =
    // dyn_cast<triton::PointerType>(offsetType.getElementType());
    // SmallVector<int64_t> strides(offsetType.getRank(), ShapedType::kDynamic);
    // auto layout = StridedLayoutAttr::get(rewriter.getContext(), 0, strides);
    // auto elemType = ptrType.getPointeeType();
    // auto memrefType = MemRefType::get(offsetType.getShape(), elemType,
    // layout);

    auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());

    ptr.dump();

    offsets.dump();

    auto memref = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, resultType.getElementType()),
        ptr);

    // Treat this as a 1-d tensor
    auto tensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                              resultType.getElementType()),
        memref, true /* restrict */, false /* writable */);

    auto emptyTensor = rewriter
                           .create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                    resultType.getElementType())
                           .getResult();

    SmallVector<AffineMap, 2> affineMaps(
        loadOp.getMask() ? 3 : 2,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));

    // auto genericOp = rewriter.create<linalg::GenericOp>(loc);

    SmallVector<Value> inputs{offsets};
    if (loadOp.getMask()) {
      inputs.push_back(loadOp.getMask());
    }

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultType}), inputs, ValueRange{emptyTensor},
        affineMaps,
        SmallVector<utils::IteratorType>(resultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          if (!loadOp.getMask()) {

            auto indexValue = args[0];
            // auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
            Value index0 = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIndexType(), indexValue);

            Value extract = rewriter.create<tensor::ExtractOp>(
                loc, tensor, ValueRange{index0});
            rewriter.create<linalg::YieldOp>(loc, extract);
          } else {
            auto mask = args[1];

            auto ifOp = rewriter.create<scf::IfOp>(
                loc, mask,
                [&](OpBuilder &b, Location loc) {
                  auto indexValue = args[0];
                  // auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
                  Value index0 = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), indexValue);

                  Value extract = rewriter.create<tensor::ExtractOp>(
                      loc, tensor, ValueRange{index0});
                  b.create<scf::YieldOp>(loc, extract);
                },
                [&](OpBuilder &b, Location loc) {
                  // TODO: Get the same type as the one from extractOp
                  Value extract;

                  if (tensor.getType().getElementType().isInteger()) {
                    extract = rewriter.create<arith::ConstantOp>(
                        loc, b.getI32IntegerAttr(0));
                  } else {
                    extract = rewriter.create<arith::ConstantOp>(
                        loc, b.getF32FloatAttr(0));
                  }
                  // b.getFloatAttr()
                  b.create<scf::YieldOp>(loc, extract);
                });

            rewriter.create<linalg::YieldOp>(loc, ifOp->getResult(0));
          }
        });

    rewriter.replaceOp(loadOp, genericOp);

    return success();
  }
};

struct StoreOpConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  StoreOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context) {}

  StoreOpConverter(MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto op =
        storeOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto results = op->getResultTypes();

    auto loc = op->getLoc();
    auto ptr = op.getInput();
    auto offsets = op.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsets.getType());

    if (!offsetType) {
      offsets.dump();
      op.dump();
      llvm::dbgs() << "offset type:\n";
      offsets.getType().dump();
      return failure();
    }

    auto resultType = dyn_cast<RankedTensorType>(storeOp.getValue().getType());

    if (auto memrefType = dyn_cast<MemRefType>(ptr.getType())) {
      if (memrefType.getRank() > 1) {
        auto indices = getReassociationIndicesForCollapse(
            memrefType.getShape(), {memrefType.getNumElements()});
        assert(indices.has_value());
        ptr = rewriter.create<memref::CollapseShapeOp>(
            loc, ptr,
            SmallVector<ReassociationIndices>{llvm::to_vector_of<int64_t>(
                llvm::index_range(0, memrefType.getRank()))});
      }
    }

    auto memref = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, resultType.getElementType()),
        ptr);

    // auto genericOp = rewriter.create<linalg::GenericOp>(loc);

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    auto ip = rewriter.saveInsertionPoint();
    SmallVector<Value> ivs;
    for (auto dim : resultType.getShape()) {
      auto ub =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dim));

      // auto forOp = rewriter.create<scf::ForOp>(loc, zero, ub, one);
      auto forOp = rewriter.create<affine::AffineForOp>(loc, 0, dim);
      ivs.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    if (!storeOp.getMask()) {
      auto offsetValue = rewriter.create<tensor::ExtractOp>(loc, offsets, ivs);
      auto storeValue =
          rewriter.create<tensor::ExtractOp>(loc, storeOp.getValue(), ivs);
      // auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);
      Value storeIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offsetValue);

      rewriter.create<memref::StoreOp>(loc, storeValue, memref, storeIndex);

    } else {
      auto maskValue =
          rewriter.create<tensor::ExtractOp>(loc, storeOp.getMask(), ivs);

      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue, false);
      rewriter.setInsertionPointToStart(
          &ifOp.getThenRegion().getBlocks().front());

      auto storeValue =
          rewriter.create<tensor::ExtractOp>(loc, storeOp.getValue(), ivs);

      auto offsetValue = rewriter.create<tensor::ExtractOp>(loc, offsets, ivs);
      Value storeIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), offsetValue);

      rewriter.create<memref::StoreOp>(loc, storeValue, memref, storeIndex);
    }

    rewriter.restoreInsertionPoint(ip);

    rewriter.eraseOp(storeOp);

    return success();
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

class TritonLoadStoreToMemrefPass
    : public TritonLoadStoreToMemrefBase<TritonLoadStoreToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }

  static bool isPtrTypeLike(Type t) {
    if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
      return isa<triton::PointerType>(tensorType.getElementType());
    }
    return isa<triton::PointerType>(t);
  }

  static Type getPointeeType(Type t) {
    if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
      if (auto pointerType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return pointerType.getPointeeType();
      }
    } else if (auto pointerType = dyn_cast<triton::PointerType>(t)) {
      return pointerType.getPointeeType();
    }
    return nullptr;
  }

  static Type getMemrefForPointer(Type t) {
    if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
      if (auto pointerType =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        return MemRefType::get(tensorType.getShape(),
                               pointerType.getPointeeType());
      }
    } else if (auto pointerType = dyn_cast<triton::PointerType>(t)) {
      return UnrankedMemRefType::get(pointerType.getPointeeType(), 0);
    }
    return nullptr;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp.walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!isPtrTypeLike(arg.getType())) {
          continue;
        }

        for (auto user : llvm::make_early_inc_range(arg.getUsers())) {
          if (auto op = dyn_cast<tts::MakeUnstructuredTensorPtrOp>(user)) {
            OpBuilder b(op);
            auto memrefType = getMemrefForPointer(arg.getType());
            auto v = b.create<UnrealizedConversionCastOp>(op->getLoc(),
                                                          memrefType, arg);
            op->setOperand(0, v.getResult(0));
            // op.setOperand(unsigned int i, Value value)
          }
        }
      }
    });

    moduleOp->dump();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    // target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<triton::LoadOp, triton::StoreOp>();

    EmptyConverter t;
    patterns.add<LoadOpConverter, ScalarLoadConverter, StoreOpConverter,
                 ScalarStoreConverter>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonLoadStoreToMemrefPass() {
  return std::make_unique<TritonLoadStoreToMemrefPass>();
}
