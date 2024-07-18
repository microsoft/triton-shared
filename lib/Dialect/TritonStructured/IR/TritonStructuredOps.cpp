#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

using namespace mlir;
using namespace mlir::tts;

namespace mlir {
namespace tts {

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

// void GetStructuredStateOp::build(OpBuilder &b, OperationState &state, Value
// ptr) {
//   SmallVector<Type> resultTypes{ptr.getType()};
//   if (auto tensorPtr = llvm::dyn_cast<RankedTensorType>(ptr.getType())) {
//     resultTypes.append(tensorPtr.getRank() * 2,
//     IndexType::get(b.getContext()));
//   }
//   build(b, state, resultTypes, ptr);
// }

} // namespace tts
} // namespace mlir
