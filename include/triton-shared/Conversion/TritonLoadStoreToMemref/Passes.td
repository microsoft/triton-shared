//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_LOAD_STORE_TO_MEMREF_CONVERSION_PASSES
#define TRITON_LOAD_STORE_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonLoadStoreToMemref : Pass<"triton-load-store-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "triton::createTritonLoadStoreToMemrefPass()";
}

#endif
