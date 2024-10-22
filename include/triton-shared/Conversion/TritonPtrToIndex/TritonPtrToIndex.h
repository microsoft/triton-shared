#ifndef TRITON_CONVERSION_TRITONPTRTOINDEX_TRITONPTRTOINDEX_H
#define TRITON_CONVERSION_TRITONPTRTOINDEX_TRITONPTRTOINDEX_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonPtrToIndexPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONPTRTOINDEX_TRITONPTRTOINDEX_H
