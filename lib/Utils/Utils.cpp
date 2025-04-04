#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}
} // namespace triton

} // namespace mlir
