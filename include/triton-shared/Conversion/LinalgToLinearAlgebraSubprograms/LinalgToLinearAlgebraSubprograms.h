#ifndef LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H
#define LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/LinalgToLinearAlgebraSubprograms/Passes.h.inc"

void populateLinalgToLinearAlgebraSubprogramsConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToLinearAlgebraSubprogramsPass();

} // namespace triton
} // namespace mlir

#endif // LINALG_TO_LINEAR_ALGEBRA_SUBPROGRAMS_H
