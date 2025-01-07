#ifndef FOLD_UNSTRUCTURED_PTR_CONVERSION_PASSES_H
#define FOLD_UNSTRUCTURED_PTR_CONVERSION_PASSES_H

#include "triton-shared/Conversion/FoldUnstructuredPtr/FoldUnstructuredPtr.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/FoldUnstructuredPtr/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
