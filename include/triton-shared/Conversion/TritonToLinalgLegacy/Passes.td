//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_LEGACY_CONVERSION_PASSES
#define TRITON_TO_LINALG_LEGACY_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgLegacy : Pass<"triton-to-linalg-legacy", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let description = [{
    This is a legacy implementation, consider using the newer triton-to-linalg which is more robust.
  }];
  let constructor = "triton::createTritonToLinalgLegacyPass()";
}

#endif
