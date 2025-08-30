#ifndef TRITON_CONVERSION_TPTR_TO_LLVM_TPTRTOLLVM_H
#define TRITON_CONVERSION_TPTR_TO_LLVM_TPTRTOLLVM_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

namespace mlir {
namespace tptr {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TPtrToLLVM/Passes.h.inc"

void populateTPtrToLLVMConversionPatterns(RewritePatternSet &patterns,
                                          TypeConverter &typeconverter);

std::unique_ptr<OperationPass<ModuleOp>> createTPtrToLLVMPass();

}  // namespace tptr
}  // namespace mlir

#endif
