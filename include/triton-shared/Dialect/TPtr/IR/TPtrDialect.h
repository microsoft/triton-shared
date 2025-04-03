#ifndef MLIR_DIALECT_TPTR_IR_TPTR_DIALECT_H_
#define MLIR_DIALECT_TPTR_IR_TPTR_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h" // Required for IR/TPtrOps.h.inc
#include "mlir/Dialect/Ptr/IR/PtrTypes.h" // Required for IR/TPtrOps.h.inc

#include "triton/Dialect/Triton/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.h.inc"

#endif
