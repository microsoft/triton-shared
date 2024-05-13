//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>

#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
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

bool PtrState::isBlockPtr() const { return !order.empty(); }

LogicalResult PtrState::addState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
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

  for (uint64_t i = 0; i < lhsState.getRank(); i++) {
    auto newOffset =
        addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
    offsets.push_back(newOffset);

    auto newStride =
        addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
    strides.push_back(newStride);

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

  for (uint64_t i = 0; i < lhs->getRank(); i++) {
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

LogicalResult PtrState::mulState(const PtrState &lhsState,
                                 const PtrState &rhsState, Operation *op,
                                 OpBuilder &builder) {
  assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

  auto loc = op->getLoc();

  // neither lhs nor rhs should have source, since multiplying base pointer
  // does not make sense
  if (lhsState.source && rhsState.source) {
    op->emitRemark("PtrAnalysis: do not support multiplying base pointers");
    return failure();
  }

  assert(!(lhsState.scalar && rhsState.scalar) &&
         "do not expect to see both lhs and rhs are scalars");

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

  for (uint64_t i = 0; i < lhs->sizes.size(); i++) {
    OpFoldResult newOffset =
        mulOFRValue(lhs->offsets[i], rhs->scalar, loc, builder);
    OpFoldResult newStride =
        mulOFRValue(lhs->strides[i], rhs->scalar, loc, builder);
    OpFoldResult newShape =
        mulOFRValue(lhs->shape[i], rhs->scalar, loc, builder);
    offsets.push_back(newOffset);
    strides.push_back(newStride);
    shape.push_back(newShape);
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

LogicalResult PtrAnalysis::visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                           const Location loc,
                                           OpBuilder &builder) {
  PtrState lhsState;
  if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
    return failure();
  }

  PtrState rhsState;
  if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed())
    return failure();

  // Checking for higher dimension is done in addState below
  if ((lhsState.getRank() == 1 && lhsState.hasModulo()) ||
      (rhsState.getRank() == 1 && rhsState.hasModulo())) {
    addOp->emitRemark(
        "PtrAnalysis: do not support this pattern: a + arange(0, K) % M");
    return failure();
  }

  return state.addState(lhsState, rhsState, addOp, builder);
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

  return state.mulState(lhsState, rhsState, mulOp, builder);
}

LogicalResult PtrAnalysis::visitOperandRem(arith::RemSIOp remOp,
                                           PtrState &state, const Location loc,
                                           OpBuilder &builder) {
  assert(state.isEmpty());

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

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    remOp->emitRemark(
        "PtrAnalysis: do not support multiple modulo within an expression");
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
    return failure();
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

  return state.addState(ptrState, offsetState, addptrOp, builder);
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

LogicalResult PtrAnalysis::visitOperand(Value operand, PtrState &state,
                                        const Location loc,
                                        OpBuilder &builder) {

  if (knownPtrs.find(operand) != knownPtrs.end()) {
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
      } else if (auto makeTensorOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        llvm_unreachable("Unexpected operand defining operation tts.make_tptr");
      } else {
        llvm_unreachable("Unexpected operand defining operation");
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
  } else {
    llvm::dbgs() << "PtrAnalysis: encountered addptr operand produced by an "
                    "unsupported operation\n";
    operand.dump();
    return failure();
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
    auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
    ptrMap.map(op.getResult(), maketptrOp.getResult());
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
      offsetValue = offset.get<Value>();
    }
    auto castOp = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), increment);
    auto mulOp = builder.create<arith::MulIOp>(loc, castOp.getResult(),
                                               stride.get<Value>());
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

LogicalResult PtrAnalysis::rewriteForOp(scf::ForOp op) {
  SmallVector<Value> newInitArgs;

  SmallVector<std::pair<int, PtrState>, 5> initArgIndexState;
  SmallVector<std::pair<int, PtrState>, 5> knownPtrsTmp;

  llvm::SmallDenseMap<int, PtrState> initArgIndexMap;

  OpBuilder builder(op);

  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
    auto mappedV = ptrMap.lookupOrNull(arg);
    PtrState state;

    if (mappedV) {
      if (auto makeTPtrOp = mappedV.getDefiningOp<tts::MakeTensorPtrOp>()) {
        if (visitOperandMakeTPtr(makeTPtrOp, state, op.getLoc(), builder)
                .succeeded()) {
          newInitArgs.push_back(mappedV);
          // Record the PtrState for later processing
          initArgIndexState.push_back(std::make_pair(i, state));
          continue;
        }
      } else if (auto makeTensorPtrOp =
                     mappedV.getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (visitOperandMakeTensorPtr(makeTensorPtrOp, state, op.getLoc(),
                                      builder)
                .succeeded()) {
          newInitArgs.push_back(mappedV);
          // Record the PtrState for later processing
          initArgIndexState.push_back(std::make_pair(i, state));
          continue;
        }
      } else if (auto addptrOp = mappedV.getDefiningOp<triton::AddPtrOp>()) {
        // We always use tt.addptr for scalar pointers. If the defininig op is
        // tt.addptr and we have a non-scalar pointer, something must have gone
        // wrong with the pass.
        assert(!isa<RankedTensorType>(addptrOp.getResult().getType()));
        if (visitOperandAddptr(addptrOp, state, op.getLoc(), builder)
                .succeeded()) {
          newInitArgs.push_back(mappedV);
          // Record the PtrState for later processing
          initArgIndexState.push_back(std::make_pair(i, state));
          continue;
        }
      }
    }
    // If any of the analysis failed, or init arg is not pointer related or
    // prior rewrite has failed. Pass as is
    newInitArgs.push_back(arg);
  }

  // For each of the PtrState recorded in the last step, insert new instructions
  // to describe offset and stride for each dimension and append them to init
  // args
  for (auto [i, state] : initArgIndexState) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the
    // end of init arg list.
    for (auto [j, s] : llvm::enumerate(state.offsets)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.offsets[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }

    for (auto [j, s] : llvm::enumerate(state.strides)) {
      auto sIntAttr = getIntAttr(s);
      if (sIntAttr) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        newInitArgs.push_back(constOp.getResult());
        state.strides[j] = constOp.getResult();
      } else {
        newInitArgs.push_back(s.get<Value>());
      }
    }

    if (state.getRank() == 0) {
      assert(state.scalar);
      // for scalar pointers, the scalar contains the offset and is the only
      // relevant state that could be updated by the loop.
      newInitArgs.push_back(state.scalar);
    }

    // Note that we want the knownPtrs to be indexed by block arg, but we
    // only have index for now. Also, the state we record is the init
    // arg, but want to to use newly created block arg. These block args
    // are not created yet. We will translate this mapping later.
    knownPtrsTmp.push_back(std::make_pair(i, state));
    levelToBlockArgIndex[level].insert(i);
  }

  // Create a new scf::ForOp that uses updated init args and same loop body
  auto newOp = builder.create<scf::ForOp>(
      op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
      newInitArgs, [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        IRMapping cloneMap;
        cloneMap.map(op.getInductionVar(), iv);
        cloneMap.map(op.getInitArgs(), newInitArgs);
        cloneMap.map(op.getRegionIterArgs(), args);

        for (auto &bodyOp : op.getRegion().getOps()) {
          b.clone(bodyOp, cloneMap);
        }
      });

  // Convert the book-keeping data structure to use the correct key and value.
  // Key is converted from init arg index to newly created block arg, and
  // Value's PtrState fields are converted from init arg to newly created block
  // arg
  int cnt = op.getRegionIterArgs().size();
  for (auto [i, state] : knownPtrsTmp) {
    for (auto it = state.offsets.begin(); it != state.offsets.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    for (auto it = state.strides.begin(); it != state.strides.end(); it++) {
      *it = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    if (state.getRank() == 0) {
      assert(state.scalar);
      state.scalar = newOp.getRegionIterArgs()[cnt];
      cnt++;
    }

    // Record the PtrState for this pointer
    auto key = newOp.getRegionIterArgs()[i];
    knownPtrs[key] = state;
    initArgIndexMap[i] = state;

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
    // used in the original IR), it will simply be removed by canonicalization.

    // For scalar pointers, there is no need to create a tts.addptr at the
    // beginning of the loop body. We don't lower tt.load and tt.store on
    // scalars in this pass; pointer arithmetics can also just use the
    // original pointer.
    if (state.getRank() != 0) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&newOp.getRegion().front());
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(key, maketptrOp.getResult());
    }
  }

  for (auto &bodyOp : newOp.getRegion().getOps()) {
    if (auto forOp = dyn_cast<scf::ForOp>(bodyOp)) {
      forOp->emitRemark("PtrAnalysis: nested loops currently not supported");
      return failure();
    }
  }

  // Update the loop body.
  if (rewriteOp(newOp).failed()) {
    newOp->erase();
    op->emitRemark(
        "PtrAnalysis: update loop body failed when rewriting for op");
    return failure();
  }

  if (op.getNumRegionIterArgs()) {
    auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
    if (rewriteYieldOp(yieldOp, initArgIndexMap).failed()) {
      newOp->erase();
      return failure();
    };
  }

  levelToBlockArgIndex.erase(level);

  // Replace only the results that correspond to the original scf.for
  auto resultsToReplaceWith = ResultRange(
      newOp.result_begin(), newOp.result_begin() + op.getNumResults());

  LLVM_DEBUG({
    llvm::dbgs() << "new for\n";
    newOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";

    llvm::dbgs() << "old for\n";
    op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  op->replaceAllUsesWith(resultsToReplaceWith);
  op->erase();

  return success();
}

LogicalResult
PtrAnalysis::rewriteYieldOp(scf::YieldOp op,
                            llvm::SmallDenseMap<int, PtrState> &knownPtrsFor) {
  if (levelToBlockArgIndex.find(level) == levelToBlockArgIndex.end()) {
    // no need to rewrite this op
    return success();
  }

  OpBuilder builder(op);

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // PtrState for those values.
  SmallVector<PtrState, 5> initArgState;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    // If this operand is not rewritten by forOp, skip
    auto thisSet = levelToBlockArgIndex.find(level)->second;
    if (thisSet.find(i) == thisSet.end())
      continue;

    auto mappedV = ptrMap.lookupOrNull(v);
    if (!mappedV) {
      op->emitRemark("Prior rewrite failure lead to yield rewrite failure");
      return failure();
    }

    PtrState state;
    LogicalResult ret = failure();
    if (auto makeTPtrOp = mappedV.getDefiningOp<tts::MakeTensorPtrOp>()) {
      ret = visitOperandMakeTPtr(makeTPtrOp, state, op.getLoc(), builder);
    } else if (auto addptrOp = mappedV.getDefiningOp<triton::AddPtrOp>()) {
      ret = visitOperandAddptr(addptrOp, state, op.getLoc(), builder);
    }
    if (ret.failed()) {
      op->emitRemark("Failed to rewrite yield op");
      return failure();
    }
    initArgState.push_back(state);

    // Verify that shape is not updated during the for loop
    auto forState = knownPtrsFor[i];
    for (auto i = 0; i < forState.getRank(); ++i) {
      if (forState.shape[i] != state.shape[i]) {
        // Special case, see comments in addState in dealing with shape/modulo
        if (i == 0 && forState.getRank() == 2) {
          if (forState.shape[1] == state.shape[0] &&
              forState.shape[0] == state.shape[1]) {
            break;
          }
        }
        assert(0);
        op->emitRemark("PtrAnalysis: operand's shape/modulo state changed "
                       "within loop body");
        return failure();
      }
    }
  }

  SmallVector<Value> operands;
  for (auto opnd : op->getOperands()) {
    auto mappedV = ptrMap.lookupOrNull(opnd);
    if (mappedV) {
      operands.push_back(mappedV);
    } else {
      operands.push_back(opnd);
    }
  }

  // For each of the PtrState recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (auto s : state.offsets) {
      if (auto sIntAttr = getIntAttr(s)) {
        auto constOp = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getIndexAttr(sIntAttr.value()));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(s.get<Value>());
      }
    }

    for (auto s : state.strides) {
      assert(!getIntAttr(s) && "PtrState strides for yield within for "
                               "loop not expected to be attribute.");
      operands.push_back(s.get<Value>());
    }

    if (state.getRank() == 0) {
      operands.push_back(state.scalar);
    }
  }

  auto newOp = builder.create<scf::YieldOp>(op->getLoc(), operands);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  op->erase();
  return success();
}

LogicalResult PtrAnalysis::rewriteLoadOp(triton::LoadOp op) {
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
  mlir::triton::MaskState mstate;
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

    scalarOther = getScalarValue(other, loc, builder);
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

LogicalResult PtrAnalysis::rewriteStoreOp(triton::StoreOp op) {
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
  mlir::triton::MaskState mstate;

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

LogicalResult PtrAnalysis::rewriteOp(Operation *rootOp) {
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
          if (rewriteLoadOp(load).failed()) {
            load->emitRemark("PtrAnalysis: Failed to rewrite LoadOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<triton::StoreOp>([&](auto store) {
          if (rewriteStoreOp(store).failed()) {
            store->emitRemark("PtrAnalysis: Failed to rewrite StoreOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Case<scf::ForOp>([&](auto forOp) {
          if (rewriteForOp(forOp).failed()) {
            forOp->emitRemark("PtrAnalysis: Failed to rewrite ForOp");
            return WalkResult::advance();
          }
          return WalkResult::skip();
        })
        .Default([&](auto) { return WalkResult::advance(); });
  });

  return success();
}

} // namespace tts
} // namespace mlir
