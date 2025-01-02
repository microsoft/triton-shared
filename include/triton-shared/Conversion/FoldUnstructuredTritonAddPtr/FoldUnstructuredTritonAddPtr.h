#ifndef TRITON_CONVERSION_FOLD_UNSTRUCTRED_TRITON_ADDPTR_FOLD_UNSTRUCTRED_TRITON_ADDPTR_H
#define TRITON_CONVERSION_FOLD_UNSTRUCTRED_TRITON_ADDPTR_FOLD_UNSTRUCTRED_TRITON_ADDPTR_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createFoldUnstructuredTritonAddPtrPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_FOLD_UNSTRUCTRED_TRITON_ADDPTR_FOLD_UNSTRUCTRED_TRITON_ADDPTR_H
