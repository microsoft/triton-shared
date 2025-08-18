#ifndef TPTR_TO_LLVM_CONVERSION_PASSES_H
#define TPTR_TO_LLVM_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"

namespace mlir {
namespace tptr {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h.inc"

}  // namespace triton
}  // namespace mlir

#endif
