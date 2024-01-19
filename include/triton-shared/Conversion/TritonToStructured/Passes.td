#ifndef TRITON_TO_STRUCTURED_CONVERSION_PASSES
#define TRITON_TO_STRUCTURED_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructured : Pass<"triton-to-structured", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createTritonToStructuredPass()";
}

#endif
