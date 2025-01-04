#ifndef LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES
#define LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToLinearAlgebraSubprograms : Pass<"linalg-to-linear-algebra-subprograms", "mlir::ModuleOp"> {
  let summary = "Convert Linalg operations to library calls";
}

#endif
