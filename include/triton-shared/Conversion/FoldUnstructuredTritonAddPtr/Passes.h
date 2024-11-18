#ifndef FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES_H
#define FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES_H

#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/FoldUnstructuredTritonAddPtr.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/FoldUnstructuredTritonAddPtr/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
