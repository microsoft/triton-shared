#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void mlir::tptr::TPtrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.cpp.inc"
      >();
}

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void mlir::tptr::TPtrDialect::initialize() {
  // addTypes<mlir::ptr::PtrType>();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.cpp.inc"

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.cpp.inc"
