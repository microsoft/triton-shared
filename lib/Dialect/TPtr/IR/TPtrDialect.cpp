#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrTypes.cpp.inc"

using namespace mlir;

namespace {
ParseResult parseIntType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getIndexType();
  return success();
}
void printIntType(OpAsmPrinter &p, Operation *op, Type ty) {
  if (!ty.isIndex())
    p << " : " << ty;
}
} // namespace

//===----------------------------------------------------------------------===//
// Dialect
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
