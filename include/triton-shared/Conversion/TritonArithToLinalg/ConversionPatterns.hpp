#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

// TODO: this should be triton-arith-to-linalg after retiring monolithic pass
#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc,
                            ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!operand.getType().dyn_cast<ShapedType>()) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = op.getValue().dyn_cast<DenseElementsAttr>()) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

static SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter) {

  auto sourceType = source.getType().cast<RankedTensorType>();
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  std::iota(std::begin(perm), std::end(perm), 0);
  std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);

  SmallVector<int64_t> transposedShape(sourceType.getShape());
  std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

//-----------------------------
// Begin of monolithic only
//-----------------------------
struct AdvanceConverter : public OpConversionPattern<triton::AdvanceOp> {
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrState pointerState;
    PtrAnalysis::rewriteAdvanceOp(op, rewriter, knownPtrs);
    return success();
  }
};

struct MakeTensorPtrConverter
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;

  void populateVectorAsIndex(SmallVector<OpFoldResult> &vec,
                             Operation::operand_range ops,
                             ConversionPatternRewriter &rewriter,
                             Location loc) const {
    for (auto opnd : ops) {
      if (opnd.getType().isa<IntegerType>()) {
        auto castOp = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), opnd);
        vec.push_back(castOp.getResult());
      } else {
        assert(opnd.getType().isa<IndexType>());
        vec.push_back(opnd);
      }
    }
  }

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    PtrState pointerState;

    auto orderSize = op.getOrder().size();
    if (orderSize > 1) {
      for (auto [first, second] :
           llvm::zip(op.getOrder().slice(0, orderSize - 2),
                     op.getOrder().slice(1, orderSize - 1))) {
        assert(first == second + 1 &&
               "Currently only support default order on block pointers");
      }
    }

    pointerState.source = rewriter.getRemappedValue(op.getBase());
    populateVectorAsIndex(pointerState.offsets, op.getOffsets(), rewriter, loc);
    populateVectorAsIndex(pointerState.strides, op.getStrides(), rewriter, loc);

    SmallVector<Value> newOffsets;
    for (auto [offset, stride] :
         llvm::zip(pointerState.offsets, pointerState.strides)) {
      auto mulOp = rewriter.create<arith::MulIOp>(loc, offset.get<Value>(),
                                                  stride.get<Value>());
      newOffsets.push_back(mulOp.getResult());
    }

    pointerState.offsets.clear();

    for (auto offset : newOffsets) {
      pointerState.offsets.push_back(offset);
    }

    ArrayRef<int64_t> resultShape;
    auto pointerType =
        op.getResult().getType().cast<mlir::triton::PointerType>();
    if (auto shapedType = pointerType.getPointeeType().dyn_cast<ShapedType>()) {
      resultShape = shapedType.getShape();
      for (auto dim_size : resultShape) {
        pointerState.sizes.push_back(
            IntegerAttr::get(IntegerType::get(op.getContext(), 64), dim_size));
      }
    } else {
      // scalar pointer, should produce a one dimensional memref
      SmallVector<int64_t> scalarShape(1, 1);
      resultShape = scalarShape;
      assert(pointerState.getRank() == 1);
    }

    auto castOp = pointerState.createCastOp(resultShape, loc, rewriter);
    rewriter.replaceOp(op, castOp.getResult());
    return success();
  }
};

struct LegacyAddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::rewriteAddptrOp(op, rewriter, knownPtrs);
    return success();
  }
};


struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

public:
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto mask = op.getMask();
    auto other = op.getOther();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar loads
    if (!op.getResult().getType().isa<ShapedType>()) {
      auto sMemRef = PtrAnalysis::getScalarMemRef(op.getPtr(), adaptor.getPtr(),
                                                  loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadOp = rewriter.create<affine::AffineLoadOp>(
          op.getLoc(), sMemRef, zeroMap, std::nullopt);
      rewriter.replaceOp(op, loadOp.getResult());
      return success();
    }

    // 1. Simple case where no mask is used.
    auto type = ptr.getType().dyn_cast<MemRefType>();
    if (!type) {
      // Seen when implicit broadcasting is done late in a chain of operations.
      // The workaround is to broadcast the pointers early in the address
      // calculation. A proper fix is complicated, but at least we can provide a
      // better error message.
      return rewriter.notifyMatchFailure(
          op, "LoadOp expects a memref, not a memref of pointers");
    }

    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    if (!mask) {
      assert(!other && "other value used in non-masked load");
      if (auto unrealizedCast =
              ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
                ModuloState::WraparoundAttr)) {

          auto memrefs = unrealizedCast.getOperands();
          auto block1 = memrefs[0];
          auto block2 = memrefs[1];

          if (wrapType.getValue().equals(ModuloState::WraparoundSideBySide)) {
            createSideBySideCopies(block1, block2, alloc, loc, rewriter);
          } else if (wrapType.getValue().equals(
                         ModuloState::WraparoundStacked)) {
            createStackedCopies(block1, block2, alloc, loc, rewriter);
          } else {
            llvm_unreachable("unexpected wraparound type");
          }
        } else {
          llvm_unreachable("unexpected unrealized cast op");
        }

      } else {
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      }

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);

      return success();
    }

    // 2. Continuous masked loads.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed()) {
      return rewriter.notifyMatchFailure(
          op, "Cannot lower continuous masked loads");
    }

    // fill load destination with other value
    if (other) {
      auto scalarOther = getScalarValue(other, loc, rewriter);
      assert(scalarOther && "other value used in masked load produced by "
                            "unsupported instruction");

      // For each dimension check if mstate.dims[i] < shape[i], or-accumulate
      // the result
      auto shape = type.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < type.getShape().size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = mstate.dims[i].dyn_cast<Value>();
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, mstate.dims[i].get<Attribute>().cast<IntegerAttr>());
        }

        auto cmpOp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmpOp.getResult())
                      .getResult();
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(
          loc, accBase, [&](OpBuilder &builder, Location loc) {
            builder.create<linalg::FillOp>(loc, ValueRange{scalarOther},
                                           ValueRange{alloc});
            builder.create<scf::YieldOp>(loc);
          });
    }

    if (auto unrealizedCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
              ModuloState::WraparoundAttr)) {

        auto memrefs = unrealizedCast.getOperands();
        auto block1 = memrefs[0];
        auto block2 = memrefs[1];

        if (wrapType.getValue().equals(ModuloState::WraparoundSideBySide)) {
          auto [subview1, subview2] =
              mstate.getSideBySideSubviews(block1, block2, loc, rewriter);

          createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
        } else if (wrapType.getValue().equals(ModuloState::WraparoundStacked)) {
          auto [subview1, subview2] =
              mstate.getStackedSubviews(block1, block2, loc, rewriter);

          createStackedCopies(subview1, subview2, alloc, loc, rewriter);
        } else {
          llvm_unreachable("unexpected wraparound type");
        }

      } else {
        llvm_unreachable("unexpected unrealized cast op");
      }

    } else {
      memref::SubViewOp srcSubview = mstate.getSubview(ptr, loc, rewriter);
      memref::SubViewOp dstSubview = mstate.getSubview(alloc, loc, rewriter);
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto val = adaptor.getValue();
    auto mask = op.getMask();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar stores
    if (!val.getType().isa<ShapedType>()) {
      auto sMemRef =
          PtrAnalysis::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      rewriter.create<affine::AffineStoreOp>(loc, val, sMemRef, zeroMap,
                                             std::nullopt);
      rewriter.eraseOp(op);
      return success();
    }

    // 1. Simple case where no mask is used.
    if (!mask) {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, val, ptr);
      storeOp.setWritable(true);
      rewriter.eraseOp(op);
      return success();
    }

    // 2. Continuous masked stores.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed())
      return failure();

    auto srcSlice = mstate.getExtractSlice(val, loc, rewriter);
    auto dstSubview = mstate.getSubview(ptr, loc, rewriter);

    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, srcSlice, dstSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);

    return success();
  }
};

struct LoopConverter : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::IndexMapSet
        levelToBlockArgIndex; // level -> set of block arg index to be replaced

    PtrAnalysis::rewriteForOp(op, rewriter, levelToBlockArgIndex, 0, knownPtrs);
    return success();
  }
};

struct YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};


// Remove all Meta ops except for AddPtr which is handled by AddPtrConverter.
// Use benefit == 10 to ensure that this pattern always takes precedence over
// other patterns.
struct MetaOpConverter : public RewritePattern {
private:
  // UseAnalysis will tag operations whose results are used only as meta-data
  // with "MetaUse" tag.
  bool isMetaUse(Operation *op) const { return op->hasAttr("MetaUse"); }

public:
  MetaOpConverter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {

    if (isa<triton::AddPtrOp>(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "AddPtrOp will be handled separately");
    }

    if (isMetaUse(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "requires meta ops");
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};


//-----------------------------
// End of monolithic only
//-----------------------------


struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = op.getType().cast<TensorType>();
    auto loc = op.getLoc();

    auto init = rewriter.create<tensor::EmptyOp>(loc, opType.getShape(),
                                                 opType.getElementType());

    auto filledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{adaptor.getSrc()},
                                    ValueRange{init})
            .result();

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) const {
    SmallVector<int64_t> broadcastDims;
    auto srcShape = src.getShape();
    auto dstShape = dst.getShape();

    for (size_t i = 0; i < srcShape.size(); i++) {
      if (dstShape[i] != srcShape[i]) {
        assert(srcShape[i] == 1);
        broadcastDims.push_back(i);
      }
    }
    assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
    return broadcastDims;
  }

  // Broadcasts input tensor based on TosaToLinalg's broadcastToShape
  AffineMap getBroadcastAffineMap(MLIRContext *context,
                                  ArrayRef<int64_t> inputShape,
                                  ArrayRef<int64_t> broadcastToShape) const {

    assert(broadcastToShape.size() >= inputShape.size());

    // Create affine map and shapes for tensor initialization.
    SmallVector<AffineExpr> outExpr;

    size_t diff = broadcastToShape.size() - inputShape.size();
    for (size_t i = 0; i < broadcastToShape.size(); i++) {
      if (i < diff) {
        continue;
      }
      size_t j = i - diff;
      if (inputShape[j] == 1) {
        // Broadcast singleton dimension
        outExpr.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      // Non-broadcast case
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }
    return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
  }

public:
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op->getNumResults() == 1 && "code assumes single result!");
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());

    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = src.getType().cast<RankedTensorType>().getRank();
    auto resType = op->getResultTypes()[0].cast<RankedTensorType>();
    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = src.getType().cast<ShapedType>().getRank();
    assert(srcRank == 2 && "only expect transposing 2D data");

    auto res = getTransposedValue(src, op.getLoc(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getResult().getType().cast<TensorType>();
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    assert(type.getShape().size() == 1 &&
           type.getElementType().getIntOrFloatBitWidth() == 32 &&
           "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
          Value res = nestedBuilder.create<arith::IndexCastOp>(
              loc, type.getElementType(), index);
          nestedBuilder.create<linalg::YieldOp>(loc, res);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct AssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condVal = op.getCondition();

    if (condVal.getType().isa<mlir::TensorType>()) {
      auto scalarVal = getScalarValue(op.getCondition(), op.getLoc(), rewriter);
      condVal = scalarVal ? scalarVal : condVal;
    }
    assert(condVal && condVal.getType().isa<mlir::IntegerType>() &&
           "Only asserts on scalars are currently supported");

    if (!condVal.getType().isInteger(1)) {
      auto zero =
          rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 32);
      auto newCond = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, condVal, zero);
      condVal = newCond.getResult();
    }

    auto assertMessage =
        llvm::formatv("{0}.py:{1}: {2} Assertion `{3}` failed", op.getFile(),
                      op.getLine(), op.getFunc(), op.getMessage());
    rewriter.create<mlir::cf::AssertOp>(op.getLoc(), condVal,
                                        assertMessage.str());

    rewriter.eraseOp(op);
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arithBitcast = rewriter.create<arith::BitcastOp>(
        op.getLoc(), op.getType(), op.getOperand());

    rewriter.replaceOp(op, arithBitcast.getResult());
    return success();
  }
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opa = adaptor.getA();
    auto opb = adaptor.getB();
    auto opc = adaptor.getC();
    auto opcOrig = op.getC();

    bool skipC = false;
    if (auto splatOp = opcOrig.getDefiningOp<triton::SplatOp>()) {
      if (auto val = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (val.getValue().cast<FloatAttr>().getValueAsDouble() == 0.) {
          skipC = true;
        }
      }
    } else if (auto constOp = opcOrig.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat() &&
            denseAttr.getSplatValue<FloatAttr>().getValueAsDouble() == 0.) {
          skipC = true;
        }
      }
    }

    auto dstType = cast<RankedTensorType>(op.getType());
    auto elemType = dstType.getElementType();
    auto loc = op.getLoc();

    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elemType);

    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), elemType, rewriter.getFloatAttr(elemType, 0));

    auto zeroes =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    auto res = rewriter
                   .create<linalg::MatmulOp>(loc, ValueRange{opa, opb},
                                             ValueRange{zeroes})
                   .getResult(0);

    if (!skipC) {
      res = rewriter.create<arith::AddFOp>(loc, res, opc);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::MaximumFOp,
               arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
               arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(
        redOp);
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::minIntN(bitWidth));
            })
            .Case([&](arith::MaxUIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return elemType.isa<FloatType>() &&
           elemType.getIntOrFloatBitWidth() <
               Float32Type::get(elemType.getContext()).getWidth() &&
           isa<arith::AddFOp>(redOp);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case([&](arith::AddFOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<arith::AddFOp>(loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::MaximumFOp, arith::MaxNumFOp,
              arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
              arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f) or addf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto isVectorReduce = sourceType.getRank() == 1;

    if (axis == sourceType.getRank() - 1 && !isVectorReduce) {
      source = getTransposedValue(source, op.getLoc(), rewriter);
      axis = sourceType.getRank() - 2;
    }

    bool convertToF32Precision = requiresF32Conversion(resType, rop);

    auto constantType = convertToF32Precision
                            ? Float32Type::get(rewriter.getContext())
                            : elemType;

    auto accBaseConstOp = getRedBaseConstOp(rewriter, rop, constantType);
    Value initTensor;

    if (isVectorReduce) {
      // The affine vectorizer cannot vectorize affine loops generated from
      // linalg.reduce for the vector reduce case, so we must rewrite the
      // linalg.reduce to affine loops manually. Here we lower to AllocTensor
      // directly instead of EmptyOp so that the subsequent pass can recognize
      // the patterns (EmptyOp is susceptible to being CSE'd away, making it
      // harder to match the patterns correctly).
      initTensor = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = rewriter.create<tensor::InsertOp>(loc, accBaseConstOp,
                                                     initTensor, ValueRange{});
    } else {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, cast<RankedTensorType>(resType).getShape(), constantType);
      initTensor = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{accBaseConstOp},
                                               ValueRange{init})
                       .result();
    }

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{source}, ValueRange{initTensor},
                SmallVector<int64_t>{axis},
                [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                  assert(inputs.size() == 2);
                  Value result =
                      getRedElement(inputs[0], inputs[1], loc, rop, opBuilder,
                                    convertToF32Precision);
                  opBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    if (sourceType.getRank() == 1) {
      finalResult =
          rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult = rewriter.create<arith::TruncFOp>(
          loc, BFloat16Type::get(rewriter.getContext()), finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        adaptor.getOperands().front().getType().cast<RankedTensorType>();
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  // We're looking for an op that looks like this:
  //
  // %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
  // ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
  // -------------------------------------------------
  // `matchTieBreakValue`                                |
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32         |
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32        |   1.
  //   %13 = arith.andi %11, %12 : i1                    |
  // -------------------------------------------------   |-> `matchShouldUpdate`
  // `matchUpdateCondition`                              |
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32         |   2.
  // -------------------------------------------------   |
  //   %15 = arith.ori %14, %13 : i1                     |
  // -------------------------------------------------
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   tt.reduce.return %16, %17 : f32, i32
  // }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  //
  // The above mlir code is lowered from this combinator in triton's
  // standard.py:
  //
  //  def _argmax_combine(value1, index1, value2, index2, tie_break_left):
  //    if tie_break_left:
  //        tie = value1 == value2 and index1 < index2
  //    else:
  //        tie = False
  //    gt = value1 > value2 or tie
  //    v_ret = core.where(gt, value1, value2)
  //    i_ret = core.where(gt, index1, index2)
  //    return v_ret, i_ret

  LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                    Value reduceValue, Value reduceIndex,
                                    mlir::Block::iterator &it,
                                    Value &tileBreakValue) const {
    // Match the following (section 1. of the above)
    //
    //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
    //   %13 = arith.andi %11, %12 : i1
    //
    // which is equivalent to the following python code
    //
    //   tie = value1 == value2 and index1 < index2

    // matching: %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (eqCmpOp) {
      if (eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ) {
        return failure();
      }
      if (currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %12 = arith.cmpi slt, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
    if (sltCmpOp) {
      if (sltCmpOp.getPredicate() != arith::CmpIPredicate::slt) {
        return failure();
      }
      if (currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %13 = arith.andi %11, %12 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto andOp = dyn_cast<arith::AndIOp>(*it++);
    if (andOp) {
      if (andOp.getLhs() != eqCmpOp || andOp.getRhs() != sltCmpOp) {
        return failure();
      }
    } else {
      return failure();
    }

    tileBreakValue = andOp;
    return success();
  }

  LogicalResult matchShouldUpdateValue(Value currValue, Value currIndex,
                                       Value reduceValue, Value reduceIndex,
                                       mlir::Block::iterator &it,
                                       Value &shouldUpdate) const {
    Value tieResult;
    if (failed(matchTieBreakResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, tieResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Tie break result match failed\n");
      return failure();
    }

    Value comparisonResult;
    if (failed(T::matchComparisonResult(currValue, currIndex, reduceValue,
                                        reduceIndex, it, comparisonResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Comparison result match failed\n");
      return failure();
    }

    // matching: %15 = arith.ori %14, %13 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto orOp = dyn_cast<arith::OrIOp>(*it++);
    if (orOp) {
      if (orOp.getLhs() != comparisonResult || orOp.getRhs() != tieResult) {
        return failure();
      }
    } else {
      return failure();
    }

    shouldUpdate = orOp;
    return success();
  }

  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue,
                      Location loc) const {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, shape, fillValue.getType());
    return rewriter
        .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                ValueRange{initTensor})
        .result();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult match(ReduceOp op) const override final {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIt = ops.begin();
    Value shouldUpdate;
    if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                      reduceIndex, opsIt, shouldUpdate))) {
      return failure();
    }

    // matching: %16 = arith.select %15, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto valueSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (valueSelectOp) {
      if (valueSelectOp.getCondition() != shouldUpdate ||
          currValue != valueSelectOp.getTrueValue() ||
          reduceValue != valueSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching:%17 = arith.select %15, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto indexSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (indexSelectOp) {
      if (indexSelectOp.getCondition() != shouldUpdate ||
          currIndex != indexSelectOp.getTrueValue() ||
          reduceIndex != indexSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: tt.reduce.return %16, %17 : f32, i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto termOp = dyn_cast<triton::ReduceReturnOp>(*opsIt++);
    if (termOp && termOp == block->getTerminator()) {
      auto opnds = termOp.getOperands();
      if (opnds != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
        return failure();
      }
    } else {
      return failure();
    }

    return success();
  }

  void rewrite(ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override final {
    auto loc = op.getLoc();

    auto elemTypes = op.getElementTypes();

    // Set the initial value of the rank-0 tensor containing
    // the result value to either -inf or +inf depending on
    // whether we're dealing with argmax or argmin
    auto valueType = elemTypes[0];
    auto valuesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, valueType,
        rewriter.getFloatAttr(valueType, T::getBaseReductionValue()));

    // Set the initial value of the rank-0 tensor containing the index of the
    // min or max value to -1
    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    // Get the shape of the resulting tensors (both for values and indices). If
    // we are reducing to a single scalar, then the result's type is a tensor of
    // rank-0, otherwise we can reuse the original result shape
    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc)};

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getOperands(), outputs,
        SmallVector<int64_t>{adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
          assert(inputs.size() == 4);

          auto tritonReduceBlock = op.getBody();
          IRMapping mapping;
          mapping.map(tritonReduceBlock->getArguments(), inputs);

          for (auto &op : tritonReduceBlock->without_terminator()) {
            b.clone(op, mapping);
          }

          auto tritonYield = tritonReduceBlock->getTerminator();
          auto results =
              llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) {
                return mapping.lookup(val);
              });
          b.create<linalg::YieldOp>(loc, results);
        });

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          rewriter.create<tensor::ExtractOp>(
              loc, valueType, linalgOp.getResults()[0], ValueRange{}),
          rewriter.create<tensor::ExtractOp>(
              loc, indexType, linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
  }
};

struct ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf ogt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return -std::numeric_limits<float>::infinity();
  }

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

struct ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf olt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return std::numeric_limits<float>::infinity();
  }

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

// get_program_id and get_num_programs:
// When launching triton kernels, we pass 6 additional arguments to indicate
// num_programs and program_id. Amongst those six, we have 3 arguments
// correspond to each axis for num_programs followed by 3 additional arguments
// for program_id.
//
// For instance, with triton kernel example_kernel(a, b, c), we have:
//  example_kernel(
//    a, b, c,
//    num_programs_axis_0,
//    num_programs_axis_1,
//    num_programs_axis_2,
//    program_id_axis_0,
//    program_id_axis_1,
//    program_id_axis_2,
//   )
//
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

private:
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  GetNumProgramsConverter(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK * 2 + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

// Convert a pair of cmpf and select to either min or max.
// Leave the pattern as simple as possible because triton has plans to emit
// min and max directly.
template <typename CmpOp>
struct MinMaxConverter : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  MinMaxConverter(MLIRContext *context)
      : OpRewritePattern<CmpOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(CmpOp cmpOp,
                                PatternRewriter &rewriter) const final {
    if (!cmpOp.getResult().hasOneUse()) {
      return failure();
    }
    auto selectOp =
        dyn_cast<arith::SelectOp>(*cmpOp.getResult().getUsers().begin());
    if (!selectOp) {
      return failure();
    }

    if (!(cmpOp.getResult() == selectOp.getCondition() &&
          cmpOp.getLhs() == selectOp.getTrueValue() &&
          cmpOp.getRhs() == selectOp.getFalseValue())) {
      return failure();
    }

    rewriteOpWithMinMax(rewriter, cmpOp, selectOp, cmpOp.getPredicate());
    rewriter.eraseOp(cmpOp);

    return success();
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpFOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpFPredicate pred) const {
    switch (pred) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpIOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<arith::MaxUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<arith::MinSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<arith::MinUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();

    auto splatConst = arith::ConstantOp::materialize(
        rewriter, attr.getSplatValue<Attribute>(), attr.getElementType(), loc);

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
        attr.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst},
                                                ValueRange{init});

    return success();
  }
};

class CumSumConverter : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  // CumSum is a specific instance of Scan that looks like the following:
  //       %1 = "tt.scan"(%0) <{axis = 1 : i32}> ({
  //       ^bb0(%arg0: f32, %arg1: f32):
  //         %2 = arith.addf %arg0, %arg1 : f32
  //         tt.scan.return %2 : f32
  //       }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  bool isCumSum(triton::ScanOp op) const {
    auto scanBlock = op.getBody();
    auto ops = llvm::map_to_vector(scanBlock->without_terminator(),
                                   [](Operation &op) { return &op; });

    if (ops.size() != 1) {
      return false;
    }

    if (auto addOp = dyn_cast<arith::AddFOp>(ops.front())) {
      if (addOp.getResult() != scanBlock->getTerminator()->getOperand(0)) {
        return false;
      }

      auto blockArgs =
          llvm::map_range(scanBlock->getArguments(), [](BlockArgument arg) {
            return dyn_cast<Value>(arg);
          });

      auto addArgs = addOp.getOperands();

      return DenseSet<Value>(blockArgs.begin(), blockArgs.end()) ==
             DenseSet<Value>(addArgs.begin(), addArgs.end());
    }

    return false;
  }

public:
  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCumSum(op)) {
      return rewriter.notifyMatchFailure(
          op, "Only support cumsum variant of scan op");
    }

    auto input = op.getOperand(0);
    auto axis = op.getAxis();
    auto type = dyn_cast<RankedTensorType>(input.getType());

    if (type.getRank() != 1 && type.getRank() != 2 &&
        axis != type.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering scan op to cumsum with rank "
              "= {1, 2} and axis = rank - 1");
    }

    Value init = rewriter.create<tensor::EmptyOp>(op.getLoc(), type.getShape(),
                                                  type.getElementType());

    rewriter.replaceOpWithNewOp<ttx::CumSumOp>(
        op, input, rewriter.getUI32IntegerAttr(axis), init);

    return success();
  }
};

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    assert(resType.isa<ShapedType>());
    auto rank = cast<RankedTensorType>(resType).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        /*numResult + numOperands*/ 3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {op.getPtr()};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              }));
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
        });
    return success();
  }
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getSrc();
    auto output = op.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType) {
      return failure();
    }
    ArrayRef<int64_t> outputShape = outputType.getShape();

    auto shape = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64TensorAttr(outputShape));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input,
                                                   shape);

    return success();
  }
};

class ExternElementwiseBinaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getLibname().compare("libdevice") || !op.getPure() ||
        op.getArgs().size() != 2)
      return failure();
#define POPULATE_BINARY_OP(FUNC_NAME, DST_OP)                                  \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getArgs()[0], op.getArgs()[1]); \
    return success();                                                          \
  }

    POPULATE_BINARY_OP("__nv_atan2f", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_atan2", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_powf", math::PowFOp);
    POPULATE_BINARY_OP("__nv_pow", math::PowFOp);

#undef POPULATE_BINARY_OP
    return failure();
  }
};

class ExternElementwiseUnaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getLibname().compare("libdevice") || !op.getPure() ||
        op.getArgs().size() != 1)
      return failure();
#define POPULATE_UNARY_OP(FUNC_NAME, DST_OP)                                   \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getArgs()[0]);                  \
    return success();                                                          \
  }

    POPULATE_UNARY_OP("__nv_fabsf", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_fabs", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_sinf", math::SinOp);
    POPULATE_UNARY_OP("__nv_sin", math::SinOp);
    POPULATE_UNARY_OP("__nv_cosf", math::CosOp);
    POPULATE_UNARY_OP("__nv_cos", math::CosOp);
    POPULATE_UNARY_OP("__nv_tanf", math::TanOp);
    POPULATE_UNARY_OP("__nv_tan", math::TanOp);
    POPULATE_UNARY_OP("__nv_logf", math::LogOp);
    POPULATE_UNARY_OP("__nv_log", math::LogOp);
    POPULATE_UNARY_OP("__nv_log10f", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log10", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log1pf", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_log1p", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_expf", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp2f", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_exp2", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_erff", math::ErfOp);
    POPULATE_UNARY_OP("__nv_erf", math::ErfOp);
    POPULATE_UNARY_OP("__nv_sqrtf", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_sqrt", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrtf", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrt", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_ceilf", math::CeilOp);
    POPULATE_UNARY_OP("__nv_ceil", math::CeilOp);
    POPULATE_UNARY_OP("__nv_floorf", math::FloorOp);
    POPULATE_UNARY_OP("__nv_floor", math::FloorOp);
    POPULATE_UNARY_OP("__nv_truncf", math::TruncOp);
    POPULATE_UNARY_OP("__nv_trunc", math::TruncOp);

#undef POPULATE_UNARY_OP
    return failure();
  }
};

static void populateExternElementwiseOpToMLIROps(RewritePatternSet &patterns) {
  patterns.add<ExternElementwiseBinaryOpConverter,
               ExternElementwiseUnaryOpConverter>(patterns.getContext());
}


} // namespace

#endif
