#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

using namespace mlir;
using namespace mlir::tts;

namespace mlir {
namespace tts {

namespace utils {
// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
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

} // namespace utils

void MakeTensorPtrOp::build(OpBuilder &b, OperationState &state, Value base,
                            ArrayRef<int64_t> sizes,
                            ArrayRef<OpFoldResult> strides,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> shape,
                            ArrayRef<int32_t> order) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();
  // non-block pointer
  if (order.empty()) {
    resType = RankedTensorType::get(sizes, basePtr);
  }
  // block pointer
  else {
    resType = triton::PointerType::get(RankedTensorType::get(sizes, elemType),
                                       basePtr.getAddressSpace());
  }

  build(b, state, resType, base, sizes, dynamicStrides, dynamicOffsets,
        dynamicShape, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), order);
}

void MakeGatherScatterTensorPtrOp::build(OpBuilder &b, OperationState &state,
                                         Value base, Value gatherScatterOffset,
                                         int gatherScatterDim,
                                         ArrayRef<int64_t> sizes,
                                         ArrayRef<OpFoldResult> strides,
                                         ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticStrides, staticOffsets;
  SmallVector<Value> dynamicStrides, dynamicOffsets;
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (i != gatherScatterDim)
      dispatchIndexOpFoldResult(offset, dynamicOffsets, staticOffsets);
    else
      staticOffsets.push_back(0);
  }
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();

  resType = triton::PointerType::get(RankedTensorType::get(sizes, elemType),
                                     basePtr.getAddressSpace());

  build(b, state, resType, base, gatherScatterOffset,
        b.getI32IntegerAttr(gatherScatterDim), b.getDenseI64ArrayAttr(sizes),
        dynamicStrides, dynamicOffsets, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets));
}

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   ArrayRef<OpFoldResult> dims, Value other) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  // non-block pointer type
  auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType());
  // block pointer type
  auto tensorPtrType = dyn_cast<triton::PointerType>(ptr.getType());

  Type resType;
  if (ptrTensorType) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resType = RankedTensorType::get(ptrTensorType.getShape(), elemType);

  } else if (tensorPtrType) {
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    resType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType());
  }
  build(b, state, resType, ptr, dynamicDims, b.getDenseI64ArrayAttr(staticDims),
        other);
}

void StoreOp::build(OpBuilder &b, OperationState &state, Value ptr, Value value,
                    ArrayRef<OpFoldResult> dims) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  build(b, state, ptr, value, dynamicDims, b.getDenseI64ArrayAttr(staticDims));
}

LogicalResult GetStructuredStateOp::verify() {
  auto expectedOffsetAndStrideTypes =
      getOffsetAndStrideTypes(getContext(), getInput().getType());

  if (!expectedOffsetAndStrideTypes.has_value()) {
    return failure();
  }

  auto [expectedOffsetTypes, expectedStrideTypes] =
      *expectedOffsetAndStrideTypes;

  return success(expectedOffsetTypes.size() == getOffsets().size() &&
                 llvm::equal(expectedOffsetTypes, getOffsets().getTypes()) &&
                 expectedStrideTypes.size() == getStrides().size() &&
                 llvm::equal(expectedStrideTypes, getStrides().getTypes()));
}

void GetStructuredStateOp::build(OpBuilder &b, OperationState &state,
                                 Value val) {
  auto type = val.getType();

  // Builder cannot fail, so we default to empty offset and stride types.
  // The invalid op will be rejected by the verifier later.
  auto [offsetTypes, strideTypes] =
      getOffsetAndStrideTypes(b.getContext(), type)
          .value_or(std::make_pair(SmallVector<Type>{}, SmallVector<Type>{}));

  build(b, state, val.getType(), offsetTypes, strideTypes, val);
}

std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
GetStructuredStateOp::getOffsetAndStrideTypes(MLIRContext *context, Type type) {
  auto sizes = getOffsetAndStrideSegmentSizes(type);
  if (!sizes.has_value()) {
    return std::nullopt;
  }
  return std::make_pair(
      SmallVector<Type>(sizes->first, IndexType::get(context)),
      SmallVector<Type>(sizes->second, IndexType::get(context)));
}

std::optional<std::pair<int32_t, int32_t>>
GetStructuredStateOp::getOffsetAndStrideSegmentSizes(Type type) {
  int32_t offsetSegmentSize = 0;
  int32_t strideSegmentSize = 0;

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    if (tensorType.getElementType().isIntOrIndex()) {
      // Tensors of offsets
      // Important note:
      // We only care about tensor of index / int (in addition to pointer type)
      // because only values of int and index type can potentially be part of a
      // pointer arithmetic sequence.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      // Unstructured pointers (tensor<!tt.ptr<type>>)
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    }
  }
  // Block pointers (!tt.ptr<tensor<type>> or !tt.ptr<type>)
  else if (auto ptrType = llvm::dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else {
      // The only relevant state that can be updated in loops for scalar
      // pointers are offset. No need to include stride here.
      offsetSegmentSize = 1;
    }
  } else {
    return std::nullopt;
  }

  return std::make_pair(offsetSegmentSize, strideSegmentSize);
}

} // namespace tts
} // namespace mlir
