#ifndef FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES
#define FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FoldUnstructuredTritonAddPtr : Pass<"fold-unstructured-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton non-block pointer to TritonStructured dialect";
  let constructor = "triton::createFoldUnstructuredTritonAddPtrPass()";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", /*default*/"32",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">
  ];
}

#endif
