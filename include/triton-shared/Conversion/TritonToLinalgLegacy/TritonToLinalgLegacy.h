//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGLEGACY_H
#define TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGLEGACY_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgLegacyPass();

void populateTritonToLinalgLegacyCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonToLinalgLegacyConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_TRITONTOLINALGLEGACY_H
