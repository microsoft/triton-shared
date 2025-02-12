#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

// using namespace mlir;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void mlir::tptr::TPtrDialect::initialize() {
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
