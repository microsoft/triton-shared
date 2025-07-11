//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "collapse-shape"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

// This pattern collapses a `linalg.fill` operation that fills a tensor with a
// single value into a `tensor.expand_shape` operation that expands a tensor
// filled with the same value to a larger shape. This is useful for optimizing
// the performance of tensor operations that involve broadcasting or filling
// large tensors with a single value.
struct CollapseFill : public OpRewritePattern<linalg::FillOp> {
  CollapseFill(MLIRContext *context)
      : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult collapseMemRef(linalg::FillOp op,
                               PatternRewriter &rewriter) const {

    auto loc = op.getLoc();
    auto result = op.getOutputs()[0];
    auto resultType = mlir::cast<MemRefType>(result.getType());
    auto rank = resultType.getRank();
    if (rank <= 1) {
      return failure();
    }
    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.push_back({});
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(i));
    }
    auto elementType = resultType.getElementType();

    auto input = rewriter.create<memref::CollapseShapeOp>(
        loc,
        MemRefType::get(llvm::ArrayRef<int64_t>{resultType.getNumElements()},
                        elementType),
        result, reassociationMap);
    op.getOutputsMutable()[0].set(input.getResult());
    return success();
  }

  LogicalResult collapseTensor(linalg::FillOp op,
                               PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto result = op.getResult(0);
    auto resultType = mlir::cast<RankedTensorType>(result.getType());
    auto rank = resultType.getRank();
    if (rank <= 1) {
      return failure();
    }
    auto elementType = resultType.getElementType();

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, llvm::ArrayRef<int64_t>{resultType.getNumElements()}, elementType);
    auto fillOp =
        rewriter.create<linalg::FillOp>(loc, op.getInputs(), ValueRange{init});

    SmallVector<ReassociationExprs> reassociationMap;
    reassociationMap.push_back({});
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(i));
    }
    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, result.getType(), fillOp.getResult(0), reassociationMap);

    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() <= 1 && "code assumes single result!");
    if (op->getNumResults() == 1) {
      return collapseTensor(op, rewriter);
    } else if (op->getNumResults() == 0) {
      return collapseMemRef(op, rewriter);
    }
    return failure();
  }
};

// This pattern collapses a `linalg.transpose` operation that transposes a
// tensor into a `tensor.collapse_shape` operation that collapses the tensor
// to a smaller shape. This is useful for optimizing the performance of tensor
// operations that involve transposing large tensors.
struct CollapseTranspose : public OpRewritePattern<linalg::TransposeOp> {
  CollapseTranspose(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getInput();
    auto sourceType = dyn_cast_or_null<RankedTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    auto elementType = sourceType.getElementType();
    if (sourceRank <= 3) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 3 for transpose collapse");
    }

    SmallVector<int64_t> perm(op.getPermutation());
    SmallVector<int64_t> transposedShape(sourceRank);
    SmallVector<ReassociationExprs> reassociationMap;
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    int dim = 0;
    SmallVector<int64_t> permIdx(sourceRank);
    for (size_t i = 0; i < sourceRank; ++i) {
      permIdx[perm[i]] = i;
    }
    // The original dim corresponds to the dim after collapse
    SmallVector<int64_t> mapDim;
    for (size_t i = 0; i < sourceRank; ++i) {
      auto id = permIdx[i];
      if (i > 0 && (id == 0 || !(perm[id] == perm[id - 1] + 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      mapDim.push_back(dim);
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    SmallVector<int64_t> newPerm;
    for (size_t i = 0; i < sourceRank; ++i) {
      if (i > 0 && newPerm.back() == mapDim[perm[i]]) {
        continue;
      }
      newPerm.push_back(mapDim[perm[i]]);
    }
    perm = newPerm;
    // update transposedShape, based on perm
    transposedShape.clear();
    for (size_t i = 0; i < perm.size(); ++i) {
      transposedShape.push_back(collapseShapeInput[perm[i]]);
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    source = rewriter.create<tensor::CollapseShapeOp>(loc, sourceType, source,
                                                      reassociationMap);

    Value transposeInit =
        rewriter.create<tensor::EmptyOp>(loc, transposedShape, elementType);

    Value transpose =
        rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
            .getResults()[0];

    SmallVector<ReassociationExprs> reassociationMapRe(reassociationMap.size());
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      for (size_t j = 0; j < reassociationMap[perm[i]].size(); ++j) {
        reassociationMapRe[i].push_back(rewriter.getAffineDimExpr(idx++));
      }
    }
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], transpose, reassociationMapRe);
    return success();
  }
};

// This pattern collapses a `linalg.generic` operation that broadcasts a tensor
// to a larger shape into a `tensor.expand_shape` operation that expands the
// tensor to the desired shape. This is useful for optimizing the performance
struct CollapseBroadCast : public OpRewritePattern<linalg::GenericOp> {
  CollapseBroadCast(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("broadcastDims")) {
      return rewriter.notifyMatchFailure(op,
                                         "expected broadcastDims attribute");
    }
    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto input = op.getInputs()[0];
    auto sourceType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    if (sourceRank <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 1 for broadcast collapse");
    }

    RankedTensorType resultType =
        cast<RankedTensorType>(op.getResultTypes()[0]);
    auto elementType = resultType.getElementType();
    // collapse input tensor from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t, 8> collapseShapeInput;
    SmallVector<int64_t, 8> collapseShapeOutput;
    SmallVector<ReassociationExprs, 8> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < sourceRank; i++) {
      if (i > 0 && !((sourceType.getDimSize(i) == 1 &&
                      sourceType.getDimSize(i - 1) == 1) ||
                     (sourceType.getDimSize(i) != 1 &&
                      sourceType.getDimSize(i - 1) != 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        collapseShapeOutput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
      collapseShapeOutput[dim] *= resultType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    input = rewriter.create<tensor::CollapseShapeOp>(loc, sourceType, input,
                                                     reassociationMap);
    resultType = RankedTensorType::get(collapseShapeOutput, elementType);
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
        loc, init->getResultTypes(), ValueRange{input}, ValueRange{init},
        indexingMaps, getNParallelLoopsAttrs(resultRank));
    rewriter.cloneRegionBefore(op.getRegion(), linalgOp.getRegion(),
                               linalgOp.getRegion().begin());
    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], linalgOp->getResult(0), reassociationMap);
    return success();
  }
};

// This pattern collapses a `linalg.reduce` operation that reduces a tensor
// to a smaller shape by summing over specified dimensions into a
// `tensor.expand_shape` operation that expands the reduced tensor to the
// desired shape. This is useful for optimizing the performance of tensor
// operations that involve reductions.
struct CollapseReduce : public OpRewritePattern<linalg::ReduceOp> {
  CollapseReduce(MLIRContext *context)
      : OpRewritePattern<linalg::ReduceOp>(context) {}
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInputs()[0];
    auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for input");
    }
    auto inputRank = inputType.getRank();
    auto dims = op.getDimensions();
    if (inputRank - dims.size() <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected input rank - reduction loops > 1 for reduce collapse");
    }
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    SmallVector<int64_t> newDims;
    SmallVector<ReassociationExprs> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < inputRank; i++) {
      bool reduceAxis = llvm::is_contained(dims, i);
      if (i > 0 && (reduceAxis || llvm::is_contained(dims, i - 1))) {
        // reduce axis
        dim++;
      }
      if (reduceAxis) {
        newDims.push_back(dim);
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim] *= inputType.getDimSize(i);
    }
    if (collapseShapeInput.size() == inputRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse reduce shape");
    }
    SmallVector<int64_t> collapseShapeOutput;
    for (size_t i = 0; i < collapseShapeInput.size(); i++) {
      if (llvm::is_contained(newDims, i)) {
        // reduce axis
        continue;
      }
      collapseShapeOutput.push_back(collapseShapeInput[i]);
    }
    auto elementType = inputType.getElementType();
    auto loc = op.getLoc();
    auto newInputType = RankedTensorType::get(collapseShapeInput, elementType);
    input = rewriter.create<tensor::CollapseShapeOp>(loc, newInputType, input,
                                                     reassociationMap);

    SmallVector<ReassociationExprs> reassociationMapOutput;
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      if (llvm::is_contained(newDims, i)) {
        continue; // skip reduce axis
      } else {
        reassociationMapOutput.push_back({});
      }
      for (size_t j = 0; j < reassociationMap[i].size(); ++j) {
        reassociationMapOutput.back().push_back(
            rewriter.getAffineDimExpr(idx++));
      }
    }
    auto init = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get(collapseShapeOutput, elementType),
        op.getInits()[0], reassociationMapOutput);
    auto newReduce = rewriter.create<linalg::ReduceOp>(
        loc, init->getResultTypes(), ValueRange{input}, ValueRange{init},
        newDims);
    rewriter.cloneRegionBefore(op.getRegion(), newReduce.getRegion(),
                               newReduce.getRegion().begin());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], newReduce->getResult(0),
        reassociationMapOutput);
    return success();
  }
};

class CollapseShapePasss : public CollapseShapeBase<CollapseShapePasss> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapseFill, CollapseBroadCast, CollapseTranspose,
                 CollapseReduce>(&getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCollapseShapePass() {
  return std::make_unique<CollapseShapePasss>();
}
