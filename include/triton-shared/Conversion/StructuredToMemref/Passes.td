#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
  let options = [
      Option<"convertArgsOnly", "convert-args-only", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">,
      Option<"noConvertArgs", "no-convert-args", "bool", /*default*/"false",
             "Only run the pre-processing pass which inserts tts.get_structured_state ops used in scf.for">
  ];
}

#endif
