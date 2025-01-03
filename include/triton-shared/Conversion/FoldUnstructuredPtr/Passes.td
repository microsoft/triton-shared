#ifndef FOLD_UNSTRUCTURED_PTR_CONVERSION_PASSES
#define FOLD_UNSTRUCTURED_PTR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FoldUnstructuredPtr : Pass<"fold-unstructured-ptr", "mlir::ModuleOp"> {
  let summary = "Transforms tt.addptr ops into offset accumulation ops";
  let constructor = "triton::createFoldUnstructuredPtrPass()";
  let options = [
      Option<"offsetBitWidth", "offset-bit-width", "size_t", /*default*/"32",
             "Bitwidth used for the starting offset of each pointer">
  ];
}

#endif
