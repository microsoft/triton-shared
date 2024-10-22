#ifndef TRITON_PTR_TO_INDEX_CONVERSION_PASSES
#define TRITON_PTR_TO_INDEX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToIndex : Pass<"triton-ptr-to-index", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createTritonPtrToIndexPass()";
  let options = [
      Option<"runPrepassOnly", "run-prepass-only", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"skipPrepass", "skip-prepass", "bool", /*default*/"false",
             "Skip the prepass">
  ];
}

#endif
