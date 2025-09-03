#ifndef TPTR_TO_LLVM_CONVERSION_PASSES
#define TPTR_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TPtrToLLVM : Pass<"tptr-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Tptr operations into LLVM";
  let dependentDialects = ["mlir::tptr::TPtrDialect", "mlir::LLVM::LLVMDialect", "mlir::ptr::PtrDialect"];
}
#endif
