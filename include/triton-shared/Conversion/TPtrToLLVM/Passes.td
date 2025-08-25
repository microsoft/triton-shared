#ifndef TPTR_TO_LLVM_CONVERSION_PASSES
#define TPTR_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TPtrToLLVM : Pass<"tptr-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Tptr operations into LLVM";
  let options = [
//       Option<"addptrToLLVM", "ptradd-to-llvm", "bool", /*default*/"true",
//              "Convert tptr.ptradd on tensors to llvm">,
//       Option<"fromMemrefToLLVM", "from-memref-to-llvm", "bool", /*default*/"true",
//              "Convert tptr.from_memref on tensors to llvm">,
//       Option<"toMemrefToLLVM", "to-memref-to-llvm", "bool", /*default*/"true",
//              "Convert tptr.to_memref on tensors to llvm">,
//       Option<"typeoffsetToConst", "typeoffset-to-const", "bool", /*default*/"true",
//              "Convert tptr.typeoffset to llvm.mlir.const">,
  ];
  let dependentDialects = ["mlir::tptr::TPtrDialect", "mlir::LLVM::LLVMDialect"];
}
#endif
