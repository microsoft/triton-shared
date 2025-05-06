#ifndef MLIR_DIALECT_TPTR_IR_TPTR_DIALECT_H_
#define MLIR_DIALECT_TPTR_IR_TPTR_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // Required for IR/TPtrOps.h.inc

#include "mlir/Dialect/Ptr/IR/PtrDialect.h" // Required for IR/TPtrOps.h.inc
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"   // Required for IR/TPtrOps.h.inc

//===----------------------------------------------------------------------===//
// Temporary Pointer Dialect Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h.inc"

// Include the auto-generated header files containing the declarations of the
// Temporary Pointer Dialect operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrAttributes.h.inc"
#endif
