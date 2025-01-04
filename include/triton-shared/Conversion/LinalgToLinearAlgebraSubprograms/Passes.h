#ifndef LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES_H
#define LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES_H

#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/LinalgToLinearAlgebraSubprograms.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
