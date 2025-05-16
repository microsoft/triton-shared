//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_TRANSFORMS_PASSES
#define TRITON_SHARED_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LoopCanonicalizer : Pass<"loop-canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations by iteratively applying the canonicalization patterns of all
    loaded dialects until either a fixpoint is reached or the maximum number of
    iterations/rewrites is exhausted. Canonicalization is best-effort and does
    not guarantee that the entire IR is in a canonical form after running this
    pass. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "mlir::createLoopCanonicalizerPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

#endif // TRITON_SHARED_TRANSFORMS_PASSES
