#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace mlir::tts;

namespace {
ParseResult parsePtrType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getType<ptr::PtrType>();
  return success();
}
void printPtrType(OpAsmPrinter &p, Operation *op, ptr::PtrType ty) {
  if (ty.getMemorySpace() != nullptr)
    p << " : " << ty;
}

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

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
