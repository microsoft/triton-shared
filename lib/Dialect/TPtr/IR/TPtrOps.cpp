#include "mlir/Interfaces/SideEffectInterfaces.h" // Required for IR/TPtrOps.h.inc
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.h.inc"

using namespace mlir;
using namespace mlir::tptr;

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable(),
                       SideEffects::DefaultResource::get());
}

OpFoldResult TypeOffsetOp::fold(FoldAdaptor adaptor) {
  return adaptor.getBaseTypeAttr();
}
