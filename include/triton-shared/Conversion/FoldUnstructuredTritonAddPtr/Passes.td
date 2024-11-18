#ifndef FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES
#define FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FoldUnstructuredTritonAddPtr : Pass<"simplify-unstructured-triton-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createFoldUnstructuredTritonAddPtrPass()";
  let options = [
      Option<"runPrepassOnly", "run-prepass-only", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"skipPrepass", "skip-prepass", "bool", /*default*/"false",
             "Skip the prepass">
  ];
}

#endif
