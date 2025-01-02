#ifndef FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES
#define FOLD_UNSTRUCTRED_TRITON_ADDPTR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FoldUnstructuredTritonAddPtr : Pass<"fold-unstructured-triton-ptr", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let constructor = "triton::createFoldUnstructuredTritonAddPtrPass()";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", /*default*/"32",
             "Bitwidth used for the starting offset of each pointer">
  ];
}

#endif
