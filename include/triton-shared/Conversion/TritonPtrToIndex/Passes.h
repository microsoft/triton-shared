#ifndef TRITON_PTR_TO_INDEX_CONVERSION_PASSES_H
#define TRITON_PTR_TO_INDEX_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonPtrToIndex/TritonPtrToIndex.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonPtrToIndex/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
