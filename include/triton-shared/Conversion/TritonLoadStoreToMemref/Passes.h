//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LOAD_STORE_TO_MEMREF_CONVERSION_PASSES_H
#define TRITON_LOAD_STORE_TO_MEMREF_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TritonLoadStoreToMemref/TritonLoadStoreToMemref.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TritonLoadStoreToMemref/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
