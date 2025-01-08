//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "unstructured-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h.inc"

namespace {

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
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!loadOp.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = loadOp->getLoc();

    auto makePtrOp =
        loadOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto basePtr = adaptor.getPtr();
    auto offset = makePtrOp.getOffset();

    Value loadIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offset);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(loadOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(loadIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    auto scalarLoadOp = rewriter.create<affine::AffineLoadOp>(
        loc, memref, zeroMap, std::nullopt);

    rewriter.replaceOp(loadOp, scalarLoadOp.getResult());

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
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!storeOp.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = storeOp->getLoc();

    auto makePtrOp =
        storeOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto basePtr = adaptor.getPtr();
    auto offset = makePtrOp.getOffset();

    Value storeIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offset);

    auto memref = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        getMemrefTypeForScalarPtr(
            cast<triton::PointerType>(storeOp.getPtr().getType()),
            rewriter.getContext()),
        basePtr, getAsOpFoldResult(storeIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto storeVal = storeOp.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    rewriter.create<affine::AffineStoreOp>(loc, storeVal, memref, zeroMap,
                                           std::nullopt);
    rewriter.eraseOp(storeOp);

    return success();
  }
};

// Lowering an unstructured load op (gather) into a linalg.generic op
struct LoadOpConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LoadOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context) {}

  LoadOpConverter(MLIRContext *context)
      : OpConversionPattern<triton::LoadOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = loadOp->getLoc();

    auto makePtrOp =
        loadOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = makePtrOp.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar load, skip processing
    if (!offsetType) {
      return failure();
    }

    auto loadResultType =
        dyn_cast<RankedTensorType>(loadOp.getResult().getType());

    // Treat the base pointer (memref) as 1D because the offsets are all
    // relative to a single base pointer (already collapsed).
    auto baseMemref = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic},
                        loadResultType.getElementType()),
        ptr);

    auto baseTensor =
        rewriter
            .create<bufferization::ToTensorOp>(
                loc,
                RankedTensorType::get(
                    SmallVector<int64_t>(1, ShapedType::kDynamic),
                    loadResultType.getElementType()),
                baseMemref, true /* restrict */, false /* writable */)
            .getResult();

    // The linalg.generic op should have the following inputs:
    // - the offset tensor
    // - an optional mask tensor if the load op contains mask
    SmallVector<Value> inputs{offsetTensor};

    if (loadOp.getMask()) {
      inputs.push_back(loadOp.getMask());
    }

    auto emptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, loadResultType.getShape(),
                                     loadResultType.getElementType())
            .getResult();

    // Affine maps for the inputs and output
    // If no mask is used, 2 affine maps are generated; one for the input offset
    // tensor, the other for the output tensor.
    // If mask is used, the first 2 maps are for the offset and mask tensors
    // while the last map is for the output tensor.
    SmallVector<AffineMap> affineMaps(
        loadOp.getMask() ? 3 : 2,
        rewriter.getMultiDimIdentityMap(loadResultType.getRank()));

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, SmallVector<Type>({loadResultType}), inputs,
        ValueRange{emptyTensor}, affineMaps,
        SmallVector<utils::IteratorType>(loadResultType.getRank(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto getValueAtIndex = [baseTensor](Value indexValue, Location loc,
                                              OpBuilder &b) -> Value {
            Value index0 =
                b.create<arith::IndexCastOp>(loc, b.getIndexType(), indexValue);

            return b.create<tensor::ExtractOp>(loc, baseTensor,
                                               ValueRange{index0});
          };

          if (!loadOp.getMask()) {
            // If there is no mask, simply extract the current element from the
            // base tensor and use it as the yield value.
            auto loadValue = getValueAtIndex(args[0], loc, rewriter);
            rewriter.create<linalg::YieldOp>(loc, loadValue);
          } else {
            // If the mask value is truthy, the current element is loaded from
            // the base tensor using its offset. Otherwise, if `other` is
            // present, yield `other`. If `other` is not present, a default
            // value of 0 is used.
            auto mask = args[1];
            auto ifOp = rewriter.create<scf::IfOp>(
                loc, mask,
                [&](OpBuilder &b, Location loc) {
                  // Truthy case, load from the index
                  auto loadValue = getValueAtIndex(args[0], loc, b);
                  b.create<scf::YieldOp>(loc, loadValue);
                },
                [&](OpBuilder &b, Location loc) {
                  // Falsy case, yield `other` or 0 as the default value
                  if (loadOp.getOther()) {
                    auto definingOp = loadOp.getOther().getDefiningOp();
                    if (auto constOp =
                            dyn_cast<arith::ConstantOp>(definingOp)) {
                      if (auto attr =
                              dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                        assert(attr.isSplat());
                        auto elemValue = attr.getSplatValue<Attribute>();
                        auto otherValue = arith::ConstantOp::materialize(
                            b, elemValue, attr.getElementType(), loc);
                        b.create<scf::YieldOp>(loc, otherValue.getResult());
                      } else {
                        llvm_unreachable("unexpected constant op");
                      }
                    } else if (auto fillOp =
                                   dyn_cast<linalg::FillOp>(definingOp)) {
                      b.create<scf::YieldOp>(loc, fillOp.value());
                    } else {
                      definingOp->dump();
                      llvm_unreachable("unexpected defining op");
                    }
                  } else {
                    auto elemType = baseTensor.getType().getElementType();
                    Value extract;
                    if (isa<IntegerType>(elemType)) {
                      extract = rewriter.create<arith::ConstantOp>(
                          loc, b.getIntegerAttr(elemType, 0));
                    } else if (isa<FloatType>(elemType)) {
                      extract = rewriter.create<arith::ConstantOp>(
                          loc, b.getFloatAttr(elemType, 0));
                    } else {
                      elemType.dump();
                      llvm_unreachable("unexpected type");
                    }
                    b.create<scf::YieldOp>(loc, extract);
                  }
                });

            rewriter.create<linalg::YieldOp>(loc, ifOp->getResult(0));
          }
        });

    rewriter.replaceOp(loadOp, genericOp);

    return success();
  }
};

// Lowering an unstructured store op (scatter) into an affine loop nest
struct StoreOpConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  StoreOpConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context) {}

  StoreOpConverter(MLIRContext *context)
      : OpConversionPattern<triton::StoreOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp->getLoc();

    auto makePtrOp =
        storeOp.getPtr().getDefiningOp<tts::MakeUnstructuredTensorPtrOp>();

    auto ptr = adaptor.getPtr();
    auto offsetTensor = makePtrOp.getOffset();
    auto offsetType = dyn_cast<ShapedType>(offsetTensor.getType());

    // This must be a scalar store, skip processing
    if (!offsetType) {
      return failure();
    }

    auto resultType = dyn_cast<RankedTensorType>(storeOp.getValue().getType());

    auto storeMemref = rewriter.create<memref::CastOp>(
        loc,
        MemRefType::get({ShapedType::kDynamic}, resultType.getElementType()),
        ptr);

    auto ip = rewriter.saveInsertionPoint();

    SmallVector<Value> ivs;
    for (auto dim : resultType.getShape()) {
      auto ub =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dim));
      auto forOp = rewriter.create<affine::AffineForOp>(loc, 0, dim);
      ivs.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    if (storeOp.getMask()) {
      // Mask case, only store the value if the mask value at `ivs` is truthy
      auto maskValue =
          rewriter.create<tensor::ExtractOp>(loc, storeOp.getMask(), ivs);

      auto ifOp = rewriter.create<scf::IfOp>(loc, maskValue,
                                             false /* withElseRegion */);

      rewriter.setInsertionPointToStart(
          &ifOp.getThenRegion().getBlocks().front());
    }

    // Generate ops to store the value at each index. Note that with masking,
    // these ops are created in the `if` block generated above.
    auto offsetValue =
        rewriter.create<tensor::ExtractOp>(loc, offsetTensor, ivs);
    auto storeValue =
        rewriter.create<tensor::ExtractOp>(loc, storeOp.getValue(), ivs);
    Value storeIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offsetValue);
    rewriter.create<memref::StoreOp>(loc, storeValue, storeMemref, storeIndex);

    // Finalize
    rewriter.eraseOp(storeOp);
    rewriter.restoreInsertionPoint(ip);
    return success();
  }
};

struct MakePtrConverter
    : public OpConversionPattern<tts::MakeUnstructuredTensorPtrOp> {
  using OpConversionPattern<
      tts::MakeUnstructuredTensorPtrOp>::OpConversionPattern;

  MakePtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<tts::MakeUnstructuredTensorPtrOp>(typeConverter,
                                                              context) {}

  MakePtrConverter(MLIRContext *context)
      : OpConversionPattern<tts::MakeUnstructuredTensorPtrOp>(context) {}

  LogicalResult
  matchAndRewrite(tts::MakeUnstructuredTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The base pointer that is used in load/store comes from
    // tts.make_unstructured_tptr's base pointer. Simply replace the op with the
    // base pointer.
    rewriter.replaceOp(op, adaptor.getBase());
    return success();
  }
};

class UnstructuredToMemrefPass
    : public UnstructuredToMemrefBase<UnstructuredToMemrefPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect>();

    target.addIllegalOp<triton::LoadOp, triton::StoreOp,
                        tts::MakeUnstructuredTensorPtrOp>();

    PtrToUnrankedMemrefConverter typeConverter;
    patterns.add<MakePtrConverter>(typeConverter, patterns.getContext());

    patterns.add<LoadOpConverter, ScalarLoadConverter, StoreOpConverter,
                 ScalarStoreConverter>(patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createUnstructuredToMemrefPass() {
  return std::make_unique<UnstructuredToMemrefPass>();
}
