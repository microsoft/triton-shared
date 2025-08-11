//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "triton-ptr-analysis"

namespace mlir {

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (isa<Attribute>(ofr) && isa<IntegerAttr>(cast<Attribute>(ofr)))
    return dyn_cast<IntegerAttr>(cast<Attribute>(ofr)).getInt();

  return std::nullopt;
}

bool hasConstZero(const OpFoldResult ofr) {
  auto intAttr = getIntAttr(ofr);
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  auto val = dyn_cast<Value>(ofr);
  assert(val);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  intAttr = getIntAttr(constOp.getValue());
  if (intAttr.has_value()) {
    if (intAttr.value() == 0) {
      return true;
    }
    return false;
  }

  return false;
}

Value ofrToIndexValue(const OpFoldResult ofr, const Location loc,
                      OpBuilder &b) {
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = b.create<arith::IndexCastOp>(loc, b.getIndexType(), val);
    }
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

Value indexTypeCast(Value v, Type targetTy, const Location loc, OpBuilder &b) {
  Type ty = v.getType();
  if (isa<IndexType>(targetTy) || isa<IndexType>(ty)) {
    assert((isa<IntegerType>(targetTy) || isa<IntegerType>(ty)) &&
           "Only cast between index type and integer type");
    return b.create<arith::IndexCastOp>(loc, targetTy, v).getResult();
  } else {
    auto targetIntTy = cast<IntegerType>(targetTy);
    auto intTy = cast<IntegerType>(ty);
    if (targetIntTy.getWidth() > intTy.getWidth())
      return b.create<arith::ExtSIOp>(loc, targetTy, v).getResult();
    else
      return b.create<arith::TruncIOp>(loc, targetTy, v).getResult();
  }
}

OpFoldResult expandOFRIndex(OpFoldResult ofr, OpFoldResult targetForTy,
                            const Location loc, OpBuilder &b) {
  if (getIntAttr(targetForTy))
    return ofr;
  Value targetValueForTy = cast<Value>(targetForTy);
  Type targetTy = targetValueForTy.getType();
  auto targetShapedTy = dyn_cast<ShapedType>(targetTy);

  Value v = dyn_cast<Value>(ofr);
  if (!v)
    v = b.create<arith::ConstantOp>(loc,
                                    cast<IntegerAttr>(cast<Attribute>(ofr)));

  Type ty = v.getType();
  if (targetTy == ty)
    return ofr;

  auto shapedTy = dyn_cast<ShapedType>(ty);
  if (targetShapedTy && !shapedTy) {
    Type targetEltTy = targetShapedTy.getElementType();
    // cast to target element type first.
    if (targetEltTy != ty)
      v = indexTypeCast(v, targetEltTy, loc, b);
    return b.create<triton::SplatOp>(loc, targetTy, v).getResult();
  } else if (targetShapedTy && shapedTy) {
    Type targetEltTy = targetShapedTy.getElementType();
    Type eltTy = shapedTy.getElementType();
    if (targetShapedTy.getShape() != shapedTy.getShape()) {
      assert(targetEltTy == eltTy &&
             "Only cast between same element type shaped types");
      // This path is for case like:
      // input_ptr + (row_indices[:, None] + row_offsets[:,None] % mod_offset) *
      //   stride_m + col_offsets[None, :] * stride_n
      // The modulo will be in shape of [ROW_SIZE, 1] while row_indices is in
      // shape of [ROW_SIZE,].
      LLVM_DEBUG({
        llvm::dbgs() << "Reshaping ";
        shapedTy.dump();
        llvm::dbgs() << " to ";
        targetShapedTy.dump();
      });
      SmallVector<Value> shapeValues;
      for (auto dim : targetShapedTy.getShape()) {
        shapeValues.push_back(
            b.create<arith::ConstantOp>(loc, b.getIndexAttr(dim)));
      }
      RankedTensorType targetShapeTensorTy = RankedTensorType::get(
          targetShapedTy.getShape().size(), b.getIndexType());
      auto shapeTensor = b.create<tensor::FromElementsOp>(
          loc, targetShapeTensorTy, shapeValues);
      return b.create<triton::ReshapeOp>(loc, targetTy, v, shapeTensor)
          .getResult();
    }
    if (isa<IndexType>(targetEltTy) || isa<IndexType>(eltTy)) {
      assert((isa<IntegerType>(targetEltTy) || isa<IntegerType>(eltTy)) &&
             "Only cast between index type and integer type");
      return b.create<arith::IndexCastOp>(loc, targetTy, v).getResult();
    } else {
      auto targetIntTy = cast<IntegerType>(targetEltTy);
      auto intTy = cast<IntegerType>(eltTy);
      if (targetIntTy.getWidth() > intTy.getWidth())
        return b.create<arith::ExtSIOp>(loc, targetTy, v).getResult();
      else
        return b.create<arith::TruncIOp>(loc, targetTy, v).getResult();
    }
  } else {
    assert(!shapedTy && "src type rank should be >= target type rank");
    return indexTypeCast(v, targetTy, loc, b);
  }
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
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
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
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto sumOp = b.create<arith::SubIOp>(loc, lhsValue, rhsValue);
  return sumOp.getResult();
}

OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsValue) {
    if (auto lhsOp = lhsValue.getDefiningOp<arith::ConstantOp>()) {
      lhsIntAttr = cast<IntegerAttr>(lhsOp.getValue()).getInt();
    }
  }
  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsValue) {
    if (auto rhsOp = rhsValue.getDefiningOp<arith::ConstantOp>()) {
      rhsIntAttr = cast<IntegerAttr>(rhsOp.getValue()).getInt();
    }
  }

  // shortcut for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }

  if (rhsIntAttr) {
    if (rhsIntAttr.value() == 0)
      return rhs;
    if (rhsIntAttr.value() == 1)
      return lhs;
  }

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() * rhsIntAttr.value());

  // otherwise, need to create instructions to calculate new attribute value
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  return b.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::min(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto minOp = b.create<arith::MinSIOp>(loc, lhsValue, rhsValue);
  return minOp.getResult();
}

OpFoldResult maxOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     const Location loc, OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(std::max(lhsIntAttr.value(), rhsIntAttr.value()));

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto maxOp = b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue);
  return maxOp.getResult();
}

OpFoldResult selectOFRs(const OpFoldResult condOFR, const OpFoldResult trueOFR,
                        const OpFoldResult falseOFR, const Location loc,
                        OpBuilder &b) {
  auto trueValue = ofrToIndexValue(trueOFR, loc, b);
  auto falseValue = ofrToIndexValue(falseOFR, loc, b);
  auto condValue = ofrToIndexValue(condOFR, loc, b);

  if (!condValue.getType().isInteger(1)) {
    assert(condValue.getDefiningOp<arith::IndexCastOp>());
    condValue = condValue.getDefiningOp<arith::IndexCastOp>().getOperand();
    assert(condValue.getType().isInteger(1));
  }

  auto selectOp =
      b.create<arith::SelectOp>(loc, condValue, trueValue, falseValue);
  return selectOp.getResult();
}

OpFoldResult compareOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                         const arith::CmpIPredicate pred,
                         const OpFoldResult trueOFR,
                         const OpFoldResult falseOFR, const Location loc,
                         OpBuilder &b) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return the result directly
  if (lhsIntAttr && rhsIntAttr) {
    switch (pred) {
    case arith::CmpIPredicate::eq:
      return *lhsIntAttr == *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::ne:
      return *lhsIntAttr != *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return *lhsIntAttr < *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return *lhsIntAttr <= *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return *lhsIntAttr > *rhsIntAttr ? trueOFR : falseOFR;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return *lhsIntAttr >= *rhsIntAttr ? trueOFR : falseOFR;
    default:
      llvm_unreachable("Unsupported predicate");
    }
  }

  auto lhsValue = ofrToIndexValue(lhs, loc, b);
  auto rhsValue = ofrToIndexValue(rhs, loc, b);

  auto cmpOp = b.create<arith::CmpIOp>(loc, pred, lhsValue, rhsValue);
  return selectOFRs(cmpOp.getResult(), trueOFR, falseOFR, loc, b);
}

} // namespace mlir
