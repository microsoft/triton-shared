//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <queue>
#include <string>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

namespace tts {

int32_t PtrState::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         shape.size() == offsets.size());
  return offsets.size();
}

bool PtrState::isEmpty() const {
  return (getRank() == 0 && !source && !scalar);
}

bool PtrState::hasModulo() const {
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimHasModulo(i)) {
      return true;
    }
  }
  return false;
}

bool PtrState::dimHasModulo(uint32_t dim) const {
  assert(
      !isBlockPtr() &&
      "Analysis should not check modulo if PtrState describes block pointer");

  assert(dim < getRank());

  auto intAttr = getIntAttr(shape[dim]);
  if (!intAttr.has_value()) {
    return true;
  }

  return intAttr.value() != 0;
}

bool isNotStructured(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  return value && isa<ShapedType>(value.getType());
}

bool PtrState::dimIsStructured(uint32_t dim) const {
  assert(dim < getRank());

  return !isNotStructured(offsets[dim]);
}

int32_t PtrState::getNonStructuredDim() const {
  SmallVector<int32_t> dims;
  for (int32_t i = 0; i < getRank(); i++) {
    if (dimIsStructured(i))
      continue;
    dims.emplace_back(i);
  }
  assert(dims.size() == 1 && "must have single non-continuous dimension");
  return dims.front();
}

bool PtrState::noStructuredDimExists() const {
  return getRank() > 0 && llvm::all_of(offsets, [](OpFoldResult offset) {
           return isNotStructured(offset);
         });
}

bool PtrState::isStructured() const {
  return llvm::all_of(
      offsets, [](OpFoldResult offset) { return !isNotStructured(offset); });
}

bool PtrState::isBlockPtr() const { return !order.empty(); }

bool isNotSingleDim(Value v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());
  if (!shapedTy)
    return false;
  auto valShape = shapedTy.getShape();

  // Make sure there are more than 1 dimensions with size > 1.
  return llvm::find_singleton<int64_t>(
             valShape,
             [](int64_t size, bool) {
               return size > 1 ? (int64_t *)size : nullptr;
             },
             false) == nullptr;
}

LogicalResult PtrState::rebuildAsUnsupportedOp(Value operand) {
  if (isNotSingleDim(operand))
    return failure();

  if (!isEmpty())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opType = cast<ShapedType>(operand.getType());
  // Skip support for pointer types which could be source of PtrState.
  // This check avoids creating a PtrState with non-structured source.
  if (isa<triton::PointerType>(opType.getElementType()))
    return failure();

  auto opShape = opType.getShape();

  // Setup state for unsupported operation.
  auto indexTy = IndexType::get(operand.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));
  for (auto size : opShape) {
    if (size == 1) {
      offsets.push_back(index0);
      strides.push_back(index0);
    } else {
      offsets.push_back(operand);
      strides.push_back(index1);
    }
    sizes.push_back(IntegerAttr::get(indexTy, APInt(64, size)));
    shape.push_back(index0);
  }
  return success();
}

LogicalResult PtrState::rebuildAsGatherScatter(Value op, int nonContinuousDim) {
  if (isNotSingleDim(op))
    return failure();
  if (nonContinuousDim >= getRank())
    return failure();

  // Scalar has been take care early.
  // Assume here must be shape type.
  auto opShape = cast<ShapedType>(op.getType()).getShape();
  // Make sure the op only contribute to nonContinuousDim by check
  // nonContinuousDim is the dimension with size > 1.
  if (opShape[nonContinuousDim] <= 1)
    return failure();

  // Setup state for nonContinuousDim.
  auto indexTy = IndexType::get(op.getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));

  offsets[nonContinuousDim] = op;
  strides[nonContinuousDim] = index1;
  shape[nonContinuousDim] = index0;
  return success();
}

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
  auto loc = op->getLoc();

  if (lhsState.source && rhsState.source) {
    op->emitRemark(
        "PtrAnalysis: do not support adding two pointer states that both "
        "have base pointers");
    return failure();
  }

  source = lhsState.source ? lhsState.source : rhsState.source;

  if (lhsState.scalar && rhsState.scalar) {
    auto addOp =
        builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
    scalar = addOp.getResult();
  } else if (lhsState.getRank() == 0) { // both lhs and rhs are scalars
    scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
  }

  if (!lhsState.isStructured() && !rhsState.isStructured()) {
    if (lhsState.getNonStructuredDim() != rhsState.getNonStructuredDim()) {
      op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                     "that have different non-continuous dimension");
      return failure();
    }
  }

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    if (lhsState.dimIsStructured(i) && rhsState.dimIsStructured(i)) {
      auto newOffset =
          addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
      offsets.push_back(newOffset);
      auto newStride =
          addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
      strides.push_back(newStride);
    } else {
      if (isAnalysisingUnstructured) {
        assert(!lhsState.hasModulo() && !rhsState.hasModulo() &&
               "should not have dimension with modulo when analysing "
               "unstructured");
        if (hasConstZero(lhsState.strides[i]) &&
            hasConstZero(lhsState.offsets[i])) {
          // If lhs is not for dim i, we can just use rhs's stride and offset.
          offsets.push_back(rhsState.offsets[i]);
          strides.push_back(rhsState.strides[i]);
        } else if (hasConstZero(rhsState.strides[i]) &&
                   hasConstZero(rhsState.offsets[i])) {
          // If rhs is not for dim i, we can just use lhs's stride and offset.
          offsets.push_back(lhsState.offsets[i]);
          strides.push_back(lhsState.strides[i]);
        } else {
          OpFoldResult lhsOffset = lhsState.offsets[i];
          OpFoldResult rhsOffset = rhsState.offsets[i];
          OpFoldResult lhsStride = lhsState.strides[i];
          OpFoldResult rhsStride = rhsState.strides[i];
          // If stride is 0 which will happen after
          // visitOperandExpandDims/visitOperandSplat, we set the stride to 1 to
          // mul it with offset.
          if (hasConstZero(lhsStride)) {
            assert(lhsState.dimIsStructured(i) &&
                   !rhsState.dimIsStructured(i) &&
                   "If lhs stride is zero, it must be structured and rhs "
                   "stride is unstructured");
            lhsStride = builder.getIndexAttr(1);
          }
          if (hasConstZero(rhsStride)) {
            assert(rhsState.dimIsStructured(i) &&
                   !lhsState.dimIsStructured(i) &&
                   "If rhs stride is zero, it must be structured and lhs "
                   "stride is unstructured");
            rhsStride = builder.getIndexAttr(1);
          }

          // If both offset and stride not equal, we merge 2 PtrStates by change
          // offset * stride into (offset * stride) * 1 where new offset is
          // offset * stride and new stride is set to 1.
          // Then we'll have strides equal as 1, and merge them as PtrState with
          // same strides.
          if (lhsOffset != rhsOffset && lhsStride != rhsStride) {
            // Expand offset since unstructured offset has tensor type.
            OpFoldResult stride =
                expandOFRIndex(lhsStride, lhsOffset, loc, builder);
            // new offset = offset * stride
            lhsOffset = mulOFRs(lhsOffset, stride, loc, builder);
            // Expand offset since unstructured offset has tensor type.
            stride = expandOFRIndex(rhsStride, rhsOffset, loc, builder);
            // new offset = offset * stride
            rhsOffset = mulOFRs(rhsOffset, stride, loc, builder);
            // Set both strides to 1.
            lhsStride = builder.getIndexAttr(1);
            rhsStride = builder.getIndexAttr(1);
          }

          if (lhsStride == rhsStride) {
            // For case like lhs_offset * stride + rhs_offset * stride, it is same as
            // (lhs_offset + rhs_offset) * stride.
            // We can just
            // add the offsets and reuse the stride like this:
            //   offsets[i] = lhsOffset + rhsOffset
            //   strides[i] = lhsStride
            // Expand structured offset since unstructured offset has tensor type.
            if (!lhsState.dimIsStructured(i)) {
              rhsOffset = expandOFRIndex(rhsOffset, lhsOffset, loc, builder);
            } else {
              lhsOffset = expandOFRIndex(lhsOffset, rhsOffset, loc, builder);
            }
            // Add offsets.
            offsets.push_back(addOFRs(lhsOffset, rhsOffset, loc, builder));
            // Reuse stride.
            strides.push_back(lhsStride);
          } else {
            // Assert that offsets are equal if strides are not equal.
            // This is because we are already forcing the strides to be
            // equal to 1 earlier for case both offsets and strides not equal.
            assert(lhsOffset == rhsOffset &&
                   "If strides are not equal, offsets must be equal");
            // For case like offset * lhs_stride + offset * rhs_stride, it is same as
            // offset * (lhs_stride + rhs_stride).
            // We can just
            // add the strides and reuse the offset like this:
            //   offsets[i] = lhsOffset
            //   strides[i] = lhsStride + rhsStride

            // Reuse offsets.
            offsets.push_back(lhsOffset);
            // Add strides.
            strides.push_back(addOFRs(lhsStride, rhsStride, loc, builder));
          }
        }
      } else {
        // Set stride to 1 when not continuous.
        strides.push_back(builder.getIndexAttr(1));
        // New offset is offset * stride.
        auto newLhsOffset = lhsState.offsets[i];
        auto newRhsOffset = rhsState.offsets[i];
        // Just propagate the unstructured offset to the result to track the
        // unstructured dimension. The real address calculation will be done
        // later in the PtrAnalysis::visitOperandAddptr.
        auto newOffset =
            lhsState.dimIsStructured(i) ? newRhsOffset : newLhsOffset;
        offsets.push_back(newOffset);
      }
    }

    sizes.push_back(lhsState.sizes[i]);
  }

  // AddPtr where both lhs and rhs containing modulo operators not supported
  if (lhsState.hasModulo() && rhsState.hasModulo()) {
    op->emitRemark("PtrAnalysis: do not support adding two pointer states "
                   "that both have modulo");
    return failure();
  }

  if (lhsState.hasModulo() || rhsState.hasModulo()) {
    // visitOperandSplat and visitOperandExpandDims should enforce below
    assert(lhsState.getRank() <= 2);
  }

  // dealing with modulo:
  // - If lhs has no modulo, skip
  // - If rhs has zero offset on dim i, we can just use lhs's modulo
  // - If i == 0 and rhs is the result of a splat, we will allow the add. This
  // is because the user may be trying to express adding a constant offset to
  // increment dim1, but pointer analysis cannot differentiate dim1 vs dim0 in
  // this case.
  // - Else, the analysis fails

  // An example for the 3rd condition above can look like:
  // %0 = tt.splat %scalar
  // %1 = tt.splat %ptr
  // %2 = tt.arange
  // %3 = arith.remsi %2, %size
  // %4 = tt.addptr %1, %3
  // %5 = tt.addptr %4, %0
  // %5 may also occur in a loop to increment %4 every iteration.

  // Note that this is not bullet-proof. E.g., broken IR can actually increment
  // dim0 while dim0 already has modulo, since Triton offsets are element-wise
  // and not in unit of lower dimensions. However, this is highly unlikely but
  // the analysis will provide wrong result. Hence we provide a warning in this
  // case.
  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (rhs->hasModulo()) {
    std::swap(lhs, rhs);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->getRank(); i++) {
    if (!lhs->dimIsStructured(i) || !rhs->dimIsStructured(i)) {
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
      continue;
    }

    if (!lhs->dimHasModulo(i)) {
      shape.push_back(lhs->shape[i]);
    } else if (hasConstZero(rhs->offsets[i])) {
      shape.push_back(lhs->shape[i]);
    } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
      shape.push_back(lhs->shape[1]);
      shape.push_back(lhs->shape[0]);
      op->emitWarning(
          "PtrAnalysis: allowing adding pointer state with modulo in dim 0 to "
          "another pointer state with offset in dim 0.\nPlease verify the "
          "operand that contains a scalar is meant to increment pointers in "
          "dim1. If that is not the case it WILL LEAD TO WRONG COMPILATION "
          "RESULTS.\n\nTo avoid this warning, use expand_dims (instead of "
          "splat) to explicitly specify which dimension contains the scalar.");
      break;
    } else {
      op->emitRemark(
          "PtrAnalysis: do not support adding to operand with modulo");
      return failure();
    }
  }

  return success();
}

void PtrState::dump() const {
  llvm::dbgs() << "PtrState: ";
  if (source) {
    llvm::dbgs() << "source: " << source << "\n";
  }
  if (scalar) {
    llvm::dbgs() << "scalar: " << scalar << "\n";
  }

  llvm::dbgs() << "offsets:\n";
  llvm::interleave(offsets, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nstrides:\n";
  llvm::interleave(strides, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nsizes:\n";
  llvm::interleave(sizes, llvm::dbgs(), "\n");
  llvm::dbgs() << "\nshape:\n";
  llvm::interleave(shape, llvm::dbgs(), "\n");
  llvm::dbgs() << "\norder:\n";
  llvm::interleave(order, llvm::dbgs(), "\n");
  if (isStructured()) {
    llvm::dbgs() << "structured\n";
  } else {
    for (int i = 0; i < getRank(); i++) {
      llvm::dbgs() << "dim " << i;
      if (dimIsStructured(i))
        llvm::dbgs() << " structured\n";
      else
        llvm::dbgs() << " not strucuted\n";
    }
  }

  llvm::dbgs() << "\n";
}

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState,
                                 bool isAnalysisingUnstructured, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.source && rhsState.source) {
    op->emitRemark("PtrAnalysis: do not support multiplying base pointers");
    return failure();
  }

  // currently do not support both tensors are effectively non-scalar
  if (!lhsState.scalar && !rhsState.scalar) {
    op->emitRemark(
        "PtrAnalysis: only support multiplying pointer states when one of "
        "them represent a scalar");
    return failure();
  }

  PtrState const *lhs = &lhsState;
  PtrState const *rhs = &rhsState;

  if (!rhs->scalar && lhs->scalar) {
    std::swap(lhs, rhs);
  }

  if (lhsState.scalar && rhsState.scalar) {
    scalar =
        builder.create<arith::MulIOp>(loc, lhsState.scalar, rhsState.scalar);
  }

  auto indexTy = IndexType::get(op->getContext());
  auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    if (lhs->dimIsStructured(i)) {
      OpFoldResult newOffset =
          mulOFRs(lhs->offsets[i], rhs->scalar, loc, builder);
      offsets.push_back(newOffset);
      OpFoldResult newStride =
          mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
      strides.push_back(newStride);
      OpFoldResult newShape = mulOFRs(lhs->shape[i], rhs->scalar, loc, builder);
      shape.push_back(newShape);
    } else {
      assert(!lhs->dimHasModulo(i) &&
             "should not have non-structured dimension with modulo");
      if (isAnalysisingUnstructured) {
        assert(!lhs->hasModulo() &&
               "should not have non-structured dimension with modulo");
        // Keep offsets as is for unstructured dimension.
        // The address calculation will be done later in structured to
        // memref pass.
        offsets.push_back(lhs->offsets[i]);
        // Mul the scalar to stride.
        OpFoldResult newStride =
            mulOFRs(lhs->strides[i], rhs->scalar, loc, builder);
        strides.push_back(newStride);
      } else {
        // Just propagate the unstructured offset to the result to track the
        // unstructured dimension. The real address calculation will be done
        // later in the PtrAnalysis::visitOperandAddptr.
        OpFoldResult newOffset = lhs->offsets[i];
        offsets.push_back(newOffset);
        // Mul the scalar to stride.
        OpFoldResult newStride = lhs->strides[i];
        strides.push_back(newStride);
      }
      // Shape is always 0 for non-structured dimension.
      shape.push_back(index0);
    }
    sizes.push_back(lhs->sizes[i]);
  }

  if (rhs->hasModulo()) {
    op->emitRemark(
        "PtrAnalysis: do not support multiplying pointer states that has "
        "modulos");
    return failure();
  }

  return success();
}

LogicalResult PtrState::mergeUnstructuredState(const PtrState &other,
                                               Operation *op) {
  if (isStructured() || other.isStructured()) {
    op->emitRemark("Expect merging pointer states both of which are "
                   "unstructured, but got structured state");
    return failure();
  }
  if (other.getNonStructuredDim() != getNonStructuredDim()) {
    op->emitRemark("PtrAnalysis: do not support merging pointer states with "
                   "different non-structured dimensions");
    return failure();
  }
  if (getRank() != other.getRank()) {
    op->emitRemark("PtrAnalysis: do not support merging pointer states with "
                   "different ranks");
    return failure();
  }
  int gatherDim = other.getNonStructuredDim();

  // Merge gatherDim data from other.
  offsets[gatherDim] = other.offsets[gatherDim];
  strides[gatherDim] = other.strides[gatherDim];
  shape[gatherDim] = other.shape[gatherDim];

  return success();
}

tts::MakeTensorPtrOp PtrState::createTTSMakeTensorPtrOp(OpBuilder &builder,
                                                        Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  auto op = builder.create<mlir::tts::MakeTensorPtrOp>(
      loc, source, staticSizes, strides, offsets, shape, order);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

tts::MakeGatherScatterTensorPtrOp
PtrState::createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder,
                                                Location loc) {
  SmallVector<int64_t> staticSizes;
  for (size_t i = 0; i < getRank(); i++) {
    auto s = getIntAttr(sizes[i]);
    assert(s.has_value());
    staticSizes.push_back(s.value());
  }

  int nonContinuousDim = getNonStructuredDim();

  Value nonContinuousOffset = cast<Value>(offsets[nonContinuousDim]);

  // Collapse nonContinuousOffset to 1D.
  auto offsetTy = cast<ShapedType>(nonContinuousOffset.getType());
  if (offsetTy.getRank() > 1) {
    SmallVector<ReassociationExprs, 4> reassociationMap(1);
    for (int i = 0; i < offsetTy.getRank(); ++i)
      reassociationMap[0].push_back(builder.getAffineDimExpr(i));

    int offsetSize = 1;
    for (int size : offsetTy.getShape())
      offsetSize *= size;

    auto collapseTy =
        RankedTensorType::get({offsetSize}, offsetTy.getElementType());
    nonContinuousOffset =
        builder
            .create<tensor::CollapseShapeOp>(
                loc, collapseTy, nonContinuousOffset, reassociationMap)
            .getResult();
    offsets[nonContinuousDim] = nonContinuousOffset;
  }
  // Generate tts::make_gather_scatter_tensor_ptr.
  auto op = builder.create<mlir::tts::MakeGatherScatterTensorPtrOp>(
      loc, source, nonContinuousOffset, nonContinuousDim, staticSizes, strides,
      offsets);
  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::make_gather_scatter_tensor_ptr:\n";
    op->dump();
  });

  return op;
}

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  // Checking for higher dimension is done in addState below
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    addOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a + arange(0, K) % M");
    return failure();
  }

  // When one state hasModulo while other state is not structured.
  // Need to clear the modulo and use the operand as offset directly.
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    // TODO: support modulo in this case.
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(addOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(addOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(lhsState, rhsState, isAnalysisingUnstructured, addOp,
                        builder);
}

LogicalResult PtrAnalysis::visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(mulOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(mulOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  // When one state hasModulo while other state is not structured.
  // Need to clear the modulo and use the operand as offset directly.
  if (!lhsState.isStructured() && rhsState.hasModulo()) {
    // TODO: support modulo in this case.
    if (!enableMakeGatherScatterTensorPtr ||
        rhsState
            .rebuildAsGatherScatter(mulOp.getRhs(),
                                    lhsState.getNonStructuredDim())
            .failed())
      return failure();
  } else if (lhsState.hasModulo() && !rhsState.isStructured()) {
    if (!enableMakeGatherScatterTensorPtr ||
        lhsState
            .rebuildAsGatherScatter(mulOp.getLhs(),
                                    rhsState.getNonStructuredDim())
            .failed())
      return failure();
  }

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.mulState(lhsState, rhsState, isAnalysisingUnstructured, mulOp,
                        builder);
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "PtrAnalysis: isAnalysisingUnstructured should only be true "
           "when enableMakeGatherScatterTensorPtr is true");
    // If we are analyzing unstructured state, just build state from current op.
    return state.rebuildAsUnsupportedOp(remOp.getResult());
  }

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    remOp->emitRemark("PtrAnalysis: only support cases when rhs of remainder "
                      "contains scalar");
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  // When lhs already not structured, just build state from current op.
  if (!state.isStructured()) {
    return state.rebuildAsGatherScatter(remOp.getResult(),
                                        state.getNonStructuredDim());
  }

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    remOp->emitRemark(
        "PtrAnalysis: do not support multiple modulo within an expression");
    // Multiple modulo ops on an expression is not supported.
    // But when the state has only one dimension, we can make it as
    // gather/scatter tensor ptr.
    if (state.getRank() == 1 && enableMakeGatherScatterTensorPtr)
      // Build the state from the current operation as an unstructured state,
      // but only when there is a single dimension involved.
      return state.rebuildAsGatherScatter(remOp.getResult(), 0);
    else
      return failure();
  }

  if (state.getRank() == 1) {
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.shape.back() = rhsState.scalar;
  } else if (state.getRank() == 2) {
    // torch inductor expands the tensor shape before applying the modulo.
    //
    // We only support either:
    // - (tl.arange(0, end)[:, None] % mod), or
    // - (tl.arange(0, end)[None, :] % mod)
    //
    // In both cases, we apply the modulo to the non-singleton dimension.
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.shape[1] = rhsState.scalar;
    } else if (shape[1] == 1) {
      state.shape[0] = rhsState.scalar;
    } else {
      remOp->emitRemark(
          "PtrAnalysis: taking modulo on a 2D tensor with no singleton "
          "dimension not supported");
      return failure();
    }
  } else {
    remOp->emitRemark("PtrAnalysis: unsupported modulo pattern");
    return failure();
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandExtSI(arith::ExtSIOp extOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());
  return visitOperand(extOp.getIn(), state, loc, builder);
}

LogicalResult PtrAnalysis::visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                                 PtrState &state, Location loc,
                                                 OpBuilder &builder) {
  assert(state.isEmpty());

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();

  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(builder.getIndexAttr(start));
  state.sizes.push_back(builder.getIndexAttr(shape[0]));
  state.strides.push_back(builder.getIndexAttr(stride));
  state.shape.push_back(builder.getIndexAttr(0));
  return success();
}

LogicalResult
PtrAnalysis::visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder) {
  assert(state.isEmpty());

  if (visitOperand(expandDimsOp.getSrc(), state, loc, builder).failed()) {
    return failure();
  }

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  state.offsets.insert(state.offsets.begin() + axis, builder.getIndexAttr(0));
  state.sizes.insert(state.sizes.begin() + axis, builder.getIndexAttr(1));
  state.strides.insert(state.strides.begin() + axis, builder.getIndexAttr(0));
  state.shape.insert(state.shape.begin() + axis, builder.getIndexAttr(0));

  if (state.hasModulo() && state.getRank() > 2) {
    expandDimsOp->emitRemark(
        "PtrAnalysis: unsupported scenario where expand_dims result "
        "has modulo and rank > 2");
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                   PtrState &state, const Location loc,
                                   OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    broadcastOp->emitRemark("PtrAnalysis: Unsupported broadcast source type");
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  for (size_t i = 0; i < dstShape.size(); i++) {
    if (srcShape[i] == dstShape[i]) {
      continue;
    } else if (srcShape[i] < dstShape[i]) {
      state.sizes[i] = builder.getIndexAttr(dstShape[i]);
    } else {
      llvm_unreachable("unexpected dimensions used in broadcast");
    }
  }
  return success();
}

LogicalResult PtrAnalysis::visitOperandSplat(triton::SplatOp splatOp,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {
  assert(state.isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (visitOperand(src, state, loc, builder).failed()) {
    return failure();
  }

  if (isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
    for (auto s : dstShape) {
      state.offsets.push_back(builder.getIndexAttr(0));
      state.sizes.push_back(builder.getIndexAttr(s));
      state.strides.push_back(builder.getIndexAttr(0));
      state.shape.push_back(builder.getIndexAttr(0));
    }
  } else {
    splatOp->emitRemark("PtrAnalysis: unsupported splat pattern");
    return failure();
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension
  if (state.scalar)
    state.offsets[0] = state.scalar;

  if (state.hasModulo() && state.getRank() > 2) {
    splatOp->emitRemark("PtrAnalysis: unsupported scenario where splat result "
                        "has modulo and rank > 2");
    return failure();
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandAddptr(triton::AddPtrOp addptrOp,
                                              PtrState &state,
                                              const Location loc,
                                              OpBuilder &builder) {
  assert(state.isEmpty());

  PtrState ptrState;
  if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder)
          .failed()) {
    // assert(0);
    return failure();
  } else if (!ptrState.source) {
    addptrOp.dump();
  }

  PtrState offsetState;
  if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(),
                   builder)
          .failed()) {
    return failure();
  }

  assert(ptrState.source && "ptr field should provide source / base pointer");

  assert(ptrState.getRank() == offsetState.getRank() &&
         "ptr and offset field should have the same rank");

  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  return state.addState(ptrState, offsetState, isAnalysisingUnstructured,
                        addptrOp, builder);
}

LogicalResult PtrAnalysis::visitOperandConstSplat(arith::ConstantOp op,
                                                  PtrState &state,
                                                  const Location loc,
                                                  OpBuilder &builder) {
  assert(state.isEmpty());
  // this condition is to handle cases where tt.broadcast and tt.splat are
  // folded
  auto attr = cast<DenseElementsAttr>(op.getValue());
  auto elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType));
  auto values = attr.getValues<IntegerAttr>();
  auto value = values[0].getValue();
  auto constAttr = builder.getIndexAttr(value.getSExtValue());
  auto constOp = arith::ConstantOp::materialize(builder, constAttr,
                                                builder.getIndexType(), loc);

  state.scalar = constOp;

  auto resultType = cast<ShapedType>(op.getResult().getType());
  for (size_t i = 0; i < resultType.getShape().size(); i++) {
    if (i == 0) {
      state.offsets.push_back(constOp.getResult());
    } else {
      state.offsets.push_back(builder.getIndexAttr(0));
    }

    state.sizes.push_back(builder.getIndexAttr(resultType.getShape()[i]));
    state.strides.push_back(builder.getIndexAttr(0));
    state.shape.push_back(builder.getIndexAttr(0));
  }

  return success();
}

LogicalResult PtrAnalysis::visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {

  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();
  state.offsets = makeTPtrOp.getMixedOffsets();
  state.sizes = makeTPtrOp.getMixedSizes();
  state.strides = makeTPtrOp.getMixedStrides();
  state.shape = makeTPtrOp.getMixedShape();
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());

  return success();
}

LogicalResult
PtrAnalysis::visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder) {
  assert(state.isEmpty());
  state.source = makeTPtrOp.getBase();

  if (makeTPtrOp.getOrder().empty()) {
    makeTPtrOp->emitRemark(
        "PtrAnalysis: expect tt.make_tensor_ptr to have order field set");
    return failure();
  }

  auto resType = cast<triton::PointerType>(makeTPtrOp.getResult().getType());
  auto pointeeType = cast<ShapedType>(resType.getPointeeType());
  auto shape = pointeeType.getShape();

  for (int64_t i = 0; i < pointeeType.getRank(); i++) {
    state.sizes.push_back(builder.getIndexAttr(shape[i]));

    auto strideCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
    state.strides.push_back(strideCst.getResult());

    auto offsetCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);

    auto scaledOffset = builder.create<arith::MulIOp>(
        loc, offsetCst.getResult(), strideCst.getResult());
    state.offsets.push_back(scaledOffset.getResult());

    auto shapeCst = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), makeTPtrOp.getShape()[i]);
    state.shape.push_back(shapeCst.getResult());
  }
  state.order = SmallVector<int32_t>(makeTPtrOp.getOrder());
  assert(state.isBlockPtr() &&
         "tt.make_tensor_ptr pointer state should describe a block pointer");

  return success();
}

LogicalResult PtrAnalysis::visitOperandForOp(scf::ForOp forOp, Value operand,
                                             PtrState &state,
                                             const Location loc,
                                             OpBuilder &builder) {

  auto it = llvm::find(forOp->getResults(), operand);
  auto index = std::distance(forOp->getResults().begin(), it);

  auto newState = getLoopResultPtrState(forOp, index);
  if (failed(newState)) {
    forOp.emitError(
        "Rewrite for-op failed. Could not find PtrState returned by "
        "the loop.");
    return failure();
  }

  state = newState.value();
  return success();
}

LogicalResult PtrAnalysis::visitOperandIntToPtr(triton::IntToPtrOp op,
                                                PtrState &state,
                                                const Location loc,
                                                OpBuilder &builder) {
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperandBitcast(triton::BitcastOp op,
                                               PtrState &state,
                                               const Location loc,
                                               OpBuilder &builder) {
  auto resType = op.getResult().getType();
  if (isa<ShapedType>(resType)) {
    return visitOperand(op.getSrc(), state, loc, builder);
  }
  state.source = op.getResult();
  return success();
}

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (isAnalysisingUnstructured) {
    assert(enableMakeGatherScatterTensorPtr &&
           "isAnalysisingUnstructured should not be true when "
           "enableMakeGatherScatterTensorPtr is false");
  }
  // Not using knownPtrs when isAnalysisingUnstructured is true.
  // This is because we are analyzing unstructured state, the data in knownPtrs
  // is not valid for unstructured state.
  if (!isAnalysisingUnstructured &&
      knownPtrs.find(operand) != knownPtrs.end()) {
    state = knownPtrs.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    OpBuilder::InsertionGuard guard(builder);
    if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
      builder.setInsertionPointAfter(operand.getDefiningOp());
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), operand);
    state.scalar = castOp.getResult();
    return success();
  } else if (isa<IndexType>(operand.getType())) {
    state.scalar = operand;
    return success();
  }

  if (isa<triton::PointerType>(operand.getType())) {
    // A scalar pointer can either be produced by AddPtrOp or a block
    // argument
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return visitOperandAddptr(cast<triton::AddPtrOp>(op), state, loc,
                                  builder);
      } else if (auto castOp = dyn_cast<triton::BitcastOp>(op)) {
        return visitOperandBitcast(castOp, state, loc, builder);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        return visitOperandIntToPtr(intToPtrOp, state, loc, builder);
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        state.source = operand;
        return success();
      } else {
        op->emitRemark("Unexpected defining op for triton pointer operand");
        return failure();
      }
    } else {
      state.source = operand;
      return success();
    }
  }

  if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return visitOperandAdd(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::MulIOp>()) {
    return visitOperandMul(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return visitOperandMakeRange(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return visitOperandBroadcast(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return visitOperandSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return visitOperandExpandDims(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return visitOperandAddptr(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return visitOperandConstSplat(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return visitOperandRem(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return visitOperandExtSI(op, state, loc, builder);
  } else if (auto op = operand.getDefiningOp<scf::ForOp>()) {
    return visitOperandForOp(op, operand, state, loc, builder);
  } else if (!operand.getDefiningOp()) {
    if (!knownPtrs.contains(operand)) {
      return failure();
    }

    // This operand must be an iter-arg of an inner-loop in a multiple-level
    // nested loop, which means its PtrState must have already been populated
    // during rewriteForOp of the parent loop.
    state = knownPtrs[operand];
    return success();
  } else {
    llvm::dbgs() << "PtrAnalysis: encountered addptr operand produced by an "
                    "unsupported operation\n";
    operand.dump();

    if (!enableMakeGatherScatterTensorPtr)
      return failure();
    return state.rebuildAsUnsupportedOp(operand);
  }
}

LogicalResult PtrAnalysis::rewriteAddptrOp(triton::AddPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  knownPtrs[op.getResult()] = state;

  if (isa<RankedTensorType>(op.getPtr().getType())) {
    if (state.isStructured()) {
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else if (enableMakeGatherScatterTensorPtr) {
      // If there is only one dimension, return failure since there are no
      // continuous dimensions.
      if (state.getRank() == 1)
        return failure();
      PtrState unstructuredState;
      // Switch to unstructured state analysis to create offsets and strides
      // for the non-structured dimension.
      // NOTE: this is the only place where we switch to unstructured state
      // analysis.
      isAnalysisingUnstructured = true;
      // Visit the operand again to calculate the offsets and strides for the
      // unstructured state.
      LogicalResult result =
          visitOperandAddptr(op, unstructuredState, op.getLoc(), builder);
      // Switch back to structured state analysis.
      isAnalysisingUnstructured = false;
      if (result.failed()) {
        op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.addptr for "
                       "unstructured state");
        return failure();
      }
      if (state.mergeUnstructuredState(unstructuredState, op).failed()) {
        op->emitRemark(
            "PtrAnalysis: Failed to merge unstructured state for tt.addptr");
        return failure();
      }
      auto maketptrOp =
          state.createTTSMakeGatherScatterTensorPtrOp(builder, op.getLoc());
      // Update knownPtrs to merged state.
      knownPtrs[op.getResult()] = state;
      ptrMap.map(op.getResult(), maketptrOp.getResult());
    } else {
      return failure();
    }
  } else {
    // record the ptr as we have visited and built up the state for this scalar
    // pointer, which may be used by rewriteForOp later.
    ptrMap.map(op.getResult(), op.getResult());
  }
  return success();
}

LogicalResult PtrAnalysis::rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op) {
  OpBuilder builder(op);

  PtrState state;
  if (visitOperandMakeTensorPtr(op, state, op.getLoc(), builder).failed()) {
    return failure();
  }

  auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), maketptrOp.getResult());
  return success();
}

LogicalResult PtrAnalysis::rewriteAdvanceOp(triton::AdvanceOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  PtrState state;
  if (visitOperand(op->getOperand(0), state, loc, builder).failed()) {
    op->emitRemark("PtrAnalysis: Failed to analyze ptr of tt.advance");
    return failure();
  }
  assert(state.isBlockPtr() &&
         "tt.advance pointer state should describe a block pointer");

  auto incrementOffsets = op.getOffsets();

  SmallVector<OpFoldResult> newOffsets;
  for (auto [increment, offset, stride] :
       llvm::zip(incrementOffsets, state.offsets, state.strides)) {
    Value offsetValue;
    if (auto offsetIntAttr = getIntAttr(offset)) {
      auto constOp = builder.create<arith::ConstantOp>(
          loc, builder.getIndexAttr(offsetIntAttr.value()));
      offsetValue = constOp.getResult();
    } else {
      offsetValue = cast<Value>(offset);
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), increment);
    auto mulOp = builder.create<arith::MulIOp>(loc, castOp.getResult(),
                                               cast<Value>(stride));
    auto addOp =
        builder.create<arith::AddIOp>(loc, mulOp.getResult(), offsetValue);
    newOffsets.push_back(addOp.getResult());
  }

  state.offsets = SmallVector<OpFoldResult>(newOffsets);

  auto newOp = state.createTTSMakeTensorPtrOp(builder, loc);
  knownPtrs[op.getResult()] = state;
  ptrMap.map(op.getResult(), newOp.getResult());
  return success();
}

static bool isPointerType(Type t) {
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensor.getElementType());
  }
  return isa<triton::PointerType>(t);
}

FailureOr<PtrState> PtrAnalysis::getLoopInitArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto ptr = forOp.getInitArgs()[index];

  // If the pointer into the scf.for was defined by tts.get_structured_state,
  // we can get the pointer state from the original pointer (the op's input):
  //
  // %ptr, %offset_1, %offset_2,..., %stride_1, %stride_2,... =
  // tts.get_structured_state %original
  // scf.for ... (%ptr) {...}
  if (auto getStateOp = ptr.getDefiningOp<tts::GetStructuredStateOp>()) {
    auto originalPtr = getStateOp->getOperand(0);
    if (knownPtrs.count(originalPtr)) {
      return knownPtrs[originalPtr];
    }
  }

  // For nested loops scenarios, a pointer in init-args can be returned from
  // another loop of the same level:
  // e.g.:
  // clang-format off
  //  %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //    %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
  //    }
  //    %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
  //      %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
  //      %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
  //      ...
  //    }
  //    ...
  //  }
  // clang-format on
  // Notice %arg8 = %23 comes from the return value of the first loop.
  if (auto forOp = ptr.getDefiningOp<scf::ForOp>()) {
    return getLoopResultPtrState(forOp, index);
  }

  // If the pointer isn't defined by tts.get_structured_state nor another loop,
  // it means the current pointer is an iterarg of the outer loop.
  // In such cases, the outer loops would have already set up the PtrState for
  // us already.
  //
  // scf.for iterargs(%ptr = %init_arg) {
  //    scf.for iterargs(%ptr1 = %ptr) {  <--- we're dealing with `%ptr1` here.
  //          ...
  //    }
  // }
  if (knownPtrs.count(ptr)) {
    assert(!ptr.getDefiningOp() && "Expect the ptr to be an iterarg");
    return knownPtrs[ptr];
  }

  return failure();
}

PtrState PtrAnalysis::reconcileLoopPtrState(
    scf::ForOp forOp, size_t iterArgIndex, const PtrState &state,
    llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal) {
  PtrState newState = state;
  int cnt = iterArgIndex + 1;
  if (newState.getRank() == 0) {
    assert(newState.scalar);
    // for scalar pointers, the scalar contains the offset and is the only
    // relevant newState that could be updated by the loop.
    newState.scalar = getReplacementVal(forOp, cnt);
  } else {
    for (auto &offset : newState.offsets) {
      offset = getReplacementVal(forOp, cnt++);
    }

    for (auto &stride : newState.strides) {
      stride = getReplacementVal(forOp, cnt++);
    }
  }

  return newState;
}

FailureOr<PtrState> PtrAnalysis::getLoopIterArgPtrState(scf::ForOp forOp,
                                                        size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }

  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op.getRegionIterArg(index); });
}

FailureOr<PtrState> PtrAnalysis::getLoopResultPtrState(scf::ForOp forOp,
                                                       size_t index) {
  auto state = getLoopInitArgPtrState(forOp, index);
  if (failed(state)) {
    return failure();
  }

  if (!state->isStructured()) {
    // Skip if the loop init arg is not structured.
    return failure();
  }
  return reconcileLoopPtrState(
      forOp, index, state.value(),
      [](scf::ForOp op, size_t index) { return op->getResult(index); });
}

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  for (auto [i, arg] : llvm::enumerate(op.getRegionIterArgs())) {
    if (!maybeStructuredArgs.contains(arg)) {
      continue;
    }

    auto state = getLoopIterArgPtrState(op, i);
    if (failed(state)) {
      // Because the maybeStructuredArgs may contain values that are not
      // considered structured by PtrAnalysis, failing to retrieve the PtrState
      // should not fail the rewrite process.
      // We emit an error for diagnostics and debugging purposes.
      op->emitWarning(
          "Rewrite for-op failed. Could not find PtrState for iter-arg index " +
          std::to_string(i));
      continue;
    }
    // Skip when no structured dimension exists
    if (state->noStructuredDimExists())
      continue;

    // Save the current init arg's PtrState
    knownPtrs[arg] = state.value();

    // For tensors of pointers, create a tts.make_tptr at the beginning of the
    // loop body that correspond to this region iter arg. In case it is used
    // by tt.load/tt.store in the loop body before pointer updates, this will
    // make sure rewriteLoadOp/rewriteStoreOp can use the analysis result.
    // E.g., given the following input (%tensor_of_ptr is a block arg):
    // scf.for (%tensor_of_ptr) {
    //   %data = tt.load %tensor_of_ptr
    //   // more operations to update %tensor_of_ptr
    // }
    // We may produce the following output:
    // scf.for (%base_ptr, %stride, %offset) {
    //   %tensor_of_ptr = tts.make_tptr(%base_ptr, %stride, %offset)
    //   %data = tts.load %tensor_of_ptr
    //   // more operations to update %offset
    // }
    // If %tensor_of_ptr is not used (i.e., %tensor_of_ptr is updated before
    // used in the original IR), it will simply be removed by
    // canonicalization.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    // Note that there can be tensor of indices in iter-arg, so we only create
    // the make_tensor_ptr op when the arg is of pointer type.
    if (isPointerType(arg.getType())) {
      if (state->getRank() != 0) {
        OpBuilder builder(op.getRegion());
        auto maketptrOp = state->createTTSMakeTensorPtrOp(builder, op.getLoc());
        ptrMap.map(arg, maketptrOp.getResult());
      }
    }
  }

  // Recursively rewrite the inner ops
  if (rewriteOp(op).failed()) {
    op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op");
    return failure();
  }

  return success();
}

LogicalResult
PtrAnalysis::rewriteGetStructuredStateOp(tts::GetStructuredStateOp op) {
  auto tritonValue = op->getOperand(0);

  // If this triton value isn't known, it means PtrAnalysis has failed to
  // analyze this pointer. In such cases, simply remap all uses of the
  // structured value back to its original triton value.
  if (!knownPtrs.contains(tritonValue)) {
    op.emitRemark(
        "Rewrite GetStructuredStateOp failed. Could not find PtrState.");
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }

  tts::PtrState state = knownPtrs[tritonValue];
  if (!state.isStructured()) {
    op.emitRemark(
        "Rewrite GetStructuredStateOp failed. PtrState is not structured.");
    op.getResult(0).replaceAllUsesWith(tritonValue);
    return failure();
  }
  Value remappedValue =
      ptrMap.contains(tritonValue) ? ptrMap.lookup(tritonValue) : tritonValue;

  SmallVector<Value> replacements{remappedValue};
  OpBuilder builder(op);

  if (state.getRank() == 0) {
    // For scalar pointers, the scalar contains the offset and is the only
    // relevant state that could be updated by the loop.
    if (state.scalar) {
      replacements.push_back(state.scalar);
    } else {
      // This operand is a pointer directly from the kernel arguments.
      // Use offset 0.
      assert(!tritonValue.getDefiningOp());
      replacements.push_back(builder.create<arith::ConstantOp>(
          op.getLoc(), builder.getIndexAttr(0)));
    }
  } else {
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        replacements.push_back(constOp.getResult());
      } else {
        replacements.push_back(cast<Value>(s));
      }
    }
  }

  op->replaceAllUsesWith(replacements);
  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op,
                                         bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();

  if (!ptr) {
    op->emitRemark("PtrAnalysis: pointer is not replace with tts.make_tptr so "
                   "loadOp cannot be rewritten");
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    op->emitRemark("PtrAnalysis: scalar loadOp will not be rewritten");
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);
  Value scalarOther;

  OpBuilder builder(op);
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      op->emitRemark("MaskAnalysis failed");
      return failure();
    }
    dims = mstate.dims;
  }

  if (other) {
    assert(mask && "other value used while no masks are specified");

    scalarOther = utils::getScalarValue(other, loc, builder);
    if (!scalarOther) {
      op->emitRemark("other value used in masked load produced by "
                     "unsupported instruction");
      return failure();
    }
  }

  auto loadOp = builder.create<tts::LoadOp>(loc, ptr, dims, scalarOther);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::load:\n";
    loadOp->dump();
  });

  op.replaceAllUsesWith(loadOp.getResult());
  op->erase();
  return success();
}

// Structured values from the TritonStructuredDialect have offsets and strides
// that might change in each loop iteration and hence will appear in an scf.for
// iter-args like so:
//
// %structured, %offsets, %strides  = tts.get_structured_state
// scf.for (%arg0 = %structured, %arg1 = %offsets, %arg2 = %strides) {
//   %a = %arg0 + 1
//   %b = %b + 2
//   scf.for (%arg1 = %b) {
//      ...
//   }
// }
//
// In `rewriteForOp`, we have to recognize such structured values in order to
// rewrite their PtrState accordingly. Previously, only values of Pointer-like
// type (e.g.: tensor<tt.ptr<>> or tt.ptr<tensor<>>), so detecting these values
// is as easy as checking the type.
//
// Now, tensor of indices could also appear in a loop's iter-arg. To reliably
// detect all such cases, we perform a BFS-like traversal of the IR where the
// sources are the results of `tts.get_structured_state`. All values that
// originate from the results of `tts.get_structured_state` are consider
// "maybeStructured". If a loop's iter-arg is considered "maybeStructured", we
// must set up their PtrState during `rewriteForOp`.
void PtrAnalysis::initializeMaybeStructuredArgs(Operation *op) {
  std::queue<Value> q;
  DenseSet<Value> visited;

  op->walk([&q, &visited](tts::GetStructuredStateOp getStateOp) {
    Value value = getStateOp->getResult(0);
    visited.insert(value);
    q.push(value);
  });

  while (!q.empty()) {
    auto v = q.front();
    q.pop();
    for (auto user : v.getUsers()) {
      // scf.for is a special case. We have 2 set of values to consider:
      // - iter-args
      // - loop results
      // for every init arg that originates from a `tts.get_structured_state`
      // op, its corresponding iter-arg and loop result will also be considered
      // "maybeStructured".
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [argIndex, arg] :
             llvm::zip(llvm::index_range(0, forOp.getInitArgs().size()),
                       forOp.getInitArgs())) {
          if (arg != v) {
            continue;
          }
          auto iterArg = forOp.getRegionIterArg(argIndex);
          auto tiedLoopRes = forOp.getTiedLoopResult(iterArg);
          SmallVector<Value> neighbors{iterArg, tiedLoopRes};
          for (auto neighbor : neighbors) {
            maybeStructuredArgs.insert(neighbor);
            if (!visited.contains(neighbor)) {
              visited.insert(neighbor);
              q.push(neighbor);
            }
          }
        }
      } else {
        for (auto res : user->getResults()) {
          if (res.getType() != v.getType()) {
            continue;
          }
          maybeStructuredArgs.insert(res);
          if (!visited.contains(res)) {
            visited.insert(res);
            q.push(res);
          }
        }
      }
    }
  }
}

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op,
                                          bool useUnsafeMask) {
  auto ptr = ptrMap.lookupOrNull(op.getPtr());
  auto val = op.getValue();
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (!ptr) {
    op->emitRemark("PtrAnalysis: pointer is not replace with tts.make_tptr so "
                   "storeOp cannot be rewritten");
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
  if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
    op->emitRemark("PtrAnalysis: scalar storeOp will not be rewritten");
    return failure();
  }

  ArrayRef<OpFoldResult> dims;
  mlir::triton::MaskState mstate(useUnsafeMask);

  OpBuilder builder(op);

  // Analyze the mask operand to determine at runtime the size of the data
  // are moving.
  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      op->emitRemark("MaskAnalysis failed");
      return failure();
    }
    dims = mstate.dims;
  }

  auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);

  LLVM_DEBUG({
    llvm::dbgs() << "creating tts::store:\n";
    storeOp->dump();
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp, bool useUnsafeMask) {
  LLVM_DEBUG({
    llvm::dbgs() << "rewriting rootOp\n";
    rootOp->dump();
  });

  rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == rootOp) {
      return WalkResult::advance();
    }
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<triton::AddPtrOp>([&](auto addptr) {
          if (rewriteAddptrOp(addptr).failed()) {
            addptr->emitRemark("PtrAnalysis: Failed to rewrite AddPtrOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::MakeTensorPtrOp>([&](auto maketptr) {
          if (rewriteMakeTensorPtrOp(maketptr).failed()) {
            maketptr->emitRemark(
                "PtrAnalysis: Failed to rewrite MakeTensorPtrOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::AdvanceOp>([&](auto advance) {
          if (rewriteAdvanceOp(advance).failed()) {
            advance->emitRemark("PtrAnalysis: Failed to rewrite AdvanceOp");
          }
          return WalkResult::advance();
        })
        .Case<triton::LoadOp>([&](auto load) {
          if (rewriteLoadOp(load, useUnsafeMask).failed()) {
            load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store, useUnsafeMask).failed()) {
            store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          // `rewriteForOp` recursively visits its children, so regardless
          // whether the rewrite succeeds or not, we need to return "skip" so
          // that the the walk does not visit the for-op's child operations
          // the second time.
          if (rewriteForOp(forOp).failed()) {
            forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp");
          }
          return WalkResult::skip();
        })
        .Case<tts::GetStructuredStateOp>(
            [&](tts::GetStructuredStateOp getStateOp) {
              // For tensor of indices potentially being used in pointer
              // arithmetic sequence, we need to manually populate the state of
              // none already exists.
              // This process is necessary because unlike triton pointers in a
              // loop which always have a `tt.addptr` that triggers the rewrite
              // process which includes generating the ops for updating offsets
              // and strides, tensor of indices only have a simple `arith.addi`
              // (or other arith ops).
              // Without visiting these ops manually, the ops to update the
              // offsets and strides would not be generated.
              auto tritonValue = getStateOp->getOperand(0);
              if (!knownPtrs.contains(tritonValue)) {
                PtrState state;
                OpBuilder b(getStateOp);
                if (succeeded(visitOperand(tritonValue, state,
                                           getStateOp->getLoc(), b)) &&
                    state.isStructured()) {
                  knownPtrs[tritonValue] = state;
                } else {
                  getStateOp->emitRemark("PtrAnalysis: Failed to populate ptr "
                                         "state for tensor of indices");
                }
              }

              return WalkResult::skip();
            })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

} // namespace tts
} // namespace mlir
