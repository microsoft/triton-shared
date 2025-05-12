//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "RegisterTritonSharedDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton-Shared test driver\n", registry));
}
