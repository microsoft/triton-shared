//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let description = [{
    This pass uses three modular sub-passes that are more robust compared
    to the legacy implementation (--triton-to-linalg-legacy).
    Using --triton-to-linalg is equivalent to the following:
      --triton-to-structured --canonicalize --triton-arith-to-linalg --structured-to-memref
  }];
  let constructor = "triton::createTritonToLinalgPass()";
}

#endif
