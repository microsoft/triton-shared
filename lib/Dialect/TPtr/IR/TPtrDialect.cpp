#include "mlir/IR/Builders.h"

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

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
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton-shared/Dialect/TPtr/IR/TPtrAttributes.cpp.inc"
      >();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.cpp.inc"
      >();
}

bool tptr::DefaultMemorySpaceAttr::isValidLoad(
    Type type, mlir::ptr::AtomicOrdering ordering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::DefaultMemorySpaceAttr::isValidStore(
    Type type, mlir::ptr::AtomicOrdering ordering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::DefaultMemorySpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp binOp, Type type, mlir::ptr::AtomicOrdering ordering,
    IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::DefaultMemorySpaceAttr::isValidAtomicXchg(
    Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::DefaultMemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::DefaultMemorySpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrAttributes.cpp.inc"

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.cpp.inc"
