#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createTritonPtrToMemrefPass()";
}

#endif
