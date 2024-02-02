//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <cassert>

#define DEBUG_TYPE "structured-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

static OpFoldResult accumulateTargetOffset(tts::MakeTensorPtrOp op,
                                           OpBuilder &b) {
  Location loc = op->getLoc();
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : op.getOffsets()) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                      ArrayRef<int64_t> resultShape,
                                      bool useDynamicStrides = false) {

  SmallVector<int64_t> staticStrides;
  if (useDynamicStrides) {
    staticStrides.append(op.getStrides().size(), ShapedType::kDynamic);
  } else {
    staticStrides.append(op.getStaticStrides().begin(),
                         op.getStaticStrides().end());
  }

  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);

  // tensor<1024x!tt.ptr<f32, 1>>
  auto ptrType = cast<triton::PointerType>(
      cast<RankedTensorType>(op.getType()).getElementType());
  auto elementType = ptrType.getPointeeType();

  return MemRefType::get(resultShape, elementType, layout);
}

static tensor::ExtractSliceOp getExtractSlice(int rank,
                                              ArrayRef<OpFoldResult> dims,
                                              Value source, const Location loc,
                                              OpBuilder &b) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

  auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets,
                                                         dims, strides);

  return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims,
                                          strides);
}

static memref::SubViewOp getSubview(int rank, ArrayRef<OpFoldResult> dims,
                                    Value source, const Location loc,
                                    OpBuilder &b) {
  auto sourceType = source.getType().cast<MemRefType>();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);

  return b.create<memref::SubViewOp>(loc, dstType.cast<MemRefType>(), source,
                                     offsets, dims, strides);
}

namespace {
struct MakeTensorPtrConverter
    : public OpConversionPattern<tts::MakeTensorPtrOp> {
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  LogicalResult rewriteSplitPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    return success();
  }

  LogicalResult rewriteRegularPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto ptr = adaptor.getBase();

    Location loc = op.getLoc();
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();

    // Accumulate final offset
    OpFoldResult targetOffset = accumulateTargetOffset(op, rewriter);

    auto resultType =
        getResultMemrefType(op, op.getStaticOffsets()[0], resultShape);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, resultType, ptr, targetOffset, op.getMixedSizes(),
        op.getMixedStrides());

    rewriter.replaceOp(op, castOp.getResult());

    return success();
  }

  LogicalResult rewriteBlockPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    return success();
  }

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::dbgs() << "hi\n";
    auto parentShape = op.getShape();
    auto order = op.getOrder();

    const bool isBlockPtr = !order.empty();
    const bool isRegularPtr =
        !isBlockPtr &&
        llvm::all_of(parentShape, [](auto size) { return size == 0; });

    if (isBlockPtr) {
      assert(0);
      // Block pointers
      return rewriteBlockPtr(op, adaptor, rewriter);
    } else if (isRegularPtr) {
      // Regular pointers because parent sizes are all 0
      return rewriteRegularPtr(op, adaptor, rewriter);
    } else {
      assert(0);
      // Split pointers
      return rewriteSplitPtr(op, adaptor, rewriter);
    }

    return success();
  }
};

struct LoadConverter : public OpConversionPattern<tts::LoadOp> {
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  LogicalResult rewriteSplitPtr(tts::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    return success();
  }

  LogicalResult rewriteRegularPtr(tts::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    return success();
  }

  LogicalResult rewriteBlockPtr(tts::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    return success();
  }

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto other = op.getOther();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(tensorType.getShape(), elemType));

    SmallVector<OpFoldResult> mixedDims = op.getMixedDims();

    // No mask
    if (mixedDims.empty()) {
      assert(!other && "other value used in non-masked load");
      rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);

      return success();
    }

    // Masked load

    // Fill load destination with other value
    if (other) {
      // For each dimension check if dims[i] < shape[i], or-accumulate
      // the result
      auto shape = tensorType.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < shape.size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = mixedDims[i].dyn_cast<Value>();
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, mixedDims[i].get<Attribute>().cast<IntegerAttr>());
        }

        Value cmp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmp);
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(
          loc, accBase, [&](OpBuilder &builder, Location loc) {
            builder.create<linalg::FillOp>(loc, ValueRange{op.getOther()},
                                           ValueRange{alloc});
            builder.create<scf::YieldOp>(loc);
          });
    }

    memref::SubViewOp srcSubview =
        getSubview(tensorType.getRank(), mixedDims, ptr, loc, rewriter);
    memref::SubViewOp dstSubview =
        getSubview(tensorType.getRank(), mixedDims, alloc, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }
};

struct StoreConverter : public OpConversionPattern<tts::StoreOp> {
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();

    auto mixedDims = op.getMixedDims();
    if (mixedDims.empty()) {
      // store with no masks
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, storeValue, ptr);
      storeOp.setWritable(true);

    } else {
      // auto srcSlice =
      //     getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      // auto dstSubview = getSubview(rank, mixedDims, ptr, loc, rewriter);

      // auto storeOp =
      // rewriter.create<bufferization::MaterializeInDestinationOp>(
      //     loc, srcSlice, dstSubview);
      // storeOp.setWritable(true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct ScalarAddptrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getType())) {
      return failure();
    }

    auto ptr = adaptor.getPtr();
    auto offset = op.getOffset();
    auto loc = op->getLoc();

    auto offsetIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), offset);

    auto elemType =
        cast<triton::PointerType>(op.getPtr().getType()).getPointeeType();
    auto memrefType = MemRefType::get({1}, elemType);

    auto castOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, memrefType, ptr, getAsOpFoldResult(offsetIndex) /*offset*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    rewriter.replaceOp(op, castOp.getResult());

    return success();
  }
};

struct ScalarLoadConverter : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto memrefPtr = adaptor.getPtr();

    memrefPtr.dump();

    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
    auto loadOp = rewriter.create<affine::AffineLoadOp>(loc, memrefPtr, zeroMap,
                                                        std::nullopt);
    rewriter.replaceOp(op, loadOp.getResult());

    return success();
  }
};

struct ScalarStoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getValue().getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto memrefPtr = adaptor.getPtr();
    auto val = op.getValue();
    auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());

    memrefPtr.dump();

    rewriter.create<affine::AffineStoreOp>(loc, val, memrefPtr, zeroMap,
                                           std::nullopt);
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

void mlir::triton::populateStructuredToMemrefConversionPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<MakeTensorPtrConverter, LoadConverter, StoreConverter,
           ScalarAddptrConverter, ScalarLoadConverter, ScalarStoreConverter>(
          patterns.getContext());
}
