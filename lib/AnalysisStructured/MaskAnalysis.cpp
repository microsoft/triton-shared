//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/AnalysisStructured/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace triton {

LogicalResult MaskSState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (operand.getType().isa<IntegerType>()) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else {
    return failure();
  }
}

LogicalResult MaskSState::addStateScalar(const MaskSState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  start = addOFRs(state.start, scalar, loc, builder);
  end = addOFRs(state.end, scalar, loc, builder);
  dims = state.dims;
  return success();
}

LogicalResult MaskSState::addStates(const MaskSState &lhsState,
                                   const MaskSState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc) << "Unexpected case where both lhs and rhs are scalars";
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unsupported scenario where neither lhs nor rhs is a scalar";
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskSState::minStates(const MaskSState &lhsState,
                                   const MaskSState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.getRank() != rhsState.getRank()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "Unexpected case where lhs and rhs have different ranks";
    return failure();
  }

  for (uint32_t i = 0; i < lhsState.getRank(); i++) {
    auto lhsDim = lhsState.dims[i];
    auto rhsDim = rhsState.dims[i];
    dims.push_back(minOFRs(lhsDim, rhsDim, loc, builder));
  }
  return success();
}

LogicalResult MaskSState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  assert(this->isEmpty());

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    assert(attr.isSplat() && elementType.isa<IntegerType>() &&
           "All elements must share a single integer constant value");
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = constOp.getValue().cast<IntegerAttr>().getInt();
    this->scalar = builder.getIndexAttr(value);
  }

  return success();
}

LogicalResult MaskSState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());
  auto castOp =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

LogicalResult MaskSState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskSState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskSState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskSState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  MaskSState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)) ||
      !lhsState.isMask())
    return failure();

  MaskSState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)) ||
      !rhsState.isMask())
    return failure();

  return this->minStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskSState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  assert(this->isEmpty());

  if (cmpOp.getPredicate() != arith::CmpIPredicate::slt) {
    InFlightDiagnostic diag = emitError(loc) << "Unsupported cmpi predicate";
    return failure();
  }

  MaskSState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder)))
    return failure();

  MaskSState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  assert((!lhsState.scalar && rhsState.scalar) && "Unsupported cmpi scenario");

  int32_t cmpDim = -1;
  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    auto dimIntAttr = getIntAttr(lhsState.dims[i]);
    if (!dimIntAttr || dimIntAttr.value() != 1) {
      if (cmpDim != -1) {
        InFlightDiagnostic diag = emitError(loc)
                                  << "Unsupported cmpi with more than one "
                                     "dimension with size larger than 1";
        return failure();
      }
      cmpDim = i;
    }
  }
  assert(cmpDim != -1 &&
         "Unexpected case where no dimension has size larger than 1");

  auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
  auto newDim = subOFRs(newEnd, lhsState.start, loc, builder);

  for (int32_t i = 0; i < lhsState.getRank(); i++) {
    if (i == cmpDim)
      this->dims.push_back(newDim);
    else
      this->dims.push_back(lhsState.dims[i]);
  }

  return success();
}

LogicalResult MaskSState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto shape = rangeOp.getType().cast<ShapedType>().getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "stride must be 1 for make_range whose result is used "
           "as load or store masks";
    return failure();
  }

  this->start = builder.getIndexAttr(start);
  this->end = builder.getIndexAttr(end);
  this->dims.push_back(builder.getIndexAttr(shape[0]));

  return success();
}

LogicalResult MaskSState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  assert(src.getType().isa<ShapedType>() &&
         "input to tt.broadcast should be a tensor");

  auto srcShape = src.getType().cast<ShapedType>().getShape();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source and destination should match");

  if (failed(parse(src, loc, builder)))
    return failure();

  for (size_t i = 0; i < srcShape.size(); i++) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i])
      this->dims[i] = builder.getIndexAttr(dstShape[i]);
    else
      llvm_unreachable("unexpected dimensions used in broadcast");
  }

  return success();
}

LogicalResult MaskSState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  assert(this->isEmpty());

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = dst.getType().cast<ShapedType>().getShape();

  if (!src.getType().isa<IntegerType>()) {
    InFlightDiagnostic diag =
        emitError(loc)
        << "splat source must be an integer scalar for load/store masks";
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  for (auto s : dstShape)
    this->dims.push_back(builder.getIndexAttr(s));

  return success();
}

LogicalResult MaskSState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  assert(this->isEmpty());

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder)))
    return failure();

  auto dstShape =
      expandDimsOp.getResult().getType().cast<ShapedType>().getShape();
  auto axis = expandDimsOp.getAxis();
  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");
  this->dims.insert(this->dims.begin() + axis, builder.getIndexAttr(1));

  return success();
}

} // namespace triton
} // namespace mlir
