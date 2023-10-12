//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && ofr.get<Attribute>().isa<IntegerAttr>())
    return ofr.get<Attribute>().dyn_cast<IntegerAttr>().getInt();

  return std::nullopt;
}

Value ofrToIndexValue(const OpFoldResult ofr, const Location loc,
                      OpBuilder &b) {
  if (Value val = ofr.dyn_cast<Value>()) {
    assert(val.getType().isIndex() && "Provided ofr is of type index");
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

SmallVector<Value> ofrsToIndexValues(ArrayRef<OpFoldResult> ofrs,
                                     const Location loc, OpBuilder &b) {
  return llvm::to_vector<4>(
      llvm::map_range(ofrs, [&](OpFoldResult ofr) -> Value {
        return ofrToIndexValue(ofr, loc, b);
      }));
}

OpFoldResult addOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;
  if (!rhsIntAttr && lhsIntAttr && lhsIntAttr.value() == 0)
    return rhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() + rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  } else {
    assert(lhsValue.getType().isa<IndexType>());
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  } else {
    assert(lhsValue.getType().isa<IndexType>());
  }

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // shortcut for special cases
  if (!lhsIntAttr && rhsIntAttr && rhsIntAttr.value() == 0)
    return lhs;

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() - rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = b.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRValue(const OpFoldResult lhs, const Value rhs,
                         const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);

  auto rhsIsConst = false;
  // if rhs is not a const, use max value since min is used to represent
  // dynamic size or stride
  auto rhsConstValue = std::numeric_limits<int64_t>::max();
  auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>();
  if (rhsOp) {
    rhsIsConst = true;
    rhsConstValue = rhsOp.getValue().cast<IntegerAttr>().getInt();
  }

  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIsConst) {
    if (rhsConstValue == 0)
      return rhsOp.getResult();
    if (rhsConstValue == 1)
      return lhs;
  }

  // 0. both lhs and rhs are constants
  if (lhsIntAttr && rhsIsConst)
    return b.getIndexAttr(lhsIntAttr.value() * rhsConstValue);

  // 1. if lhs is constant but rhs is not
  if (lhsIntAttr && !rhsIsConst) {
    auto lhsConstOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    auto mulOp = b.create<arith::MulIOp>(loc, lhsConstOp.getResult(), rhs);
    return mulOp.getResult();
  }

  // 2. if lhs is not constant
  assert(!lhsIntAttr);
  auto mulOp = b.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs);
  return mulOp.getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = lhs.dyn_cast<Value>();
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = rhs.dyn_cast<Value>();
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = b.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

} // namespace mlir
