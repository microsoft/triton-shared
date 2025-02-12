#ifndef TPTR_DIALECT
#define TPTR_DIALECT

include "triton/Dialect/Triton/IR/TritonTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def TPtr_Dialect : Dialect {
  let name = "tptr";

  let cppNamespace = "::mlir::tptr";

  let summary = "Structured Triton operations";

  let description = [{
    Typed Pointer Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::ptr::PtrDialect"
  ];

  let usePropertiesForAttributes = 1;
}

class TPtrTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TPtr_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

// Pointer Type in C++ (corresponding to `TT_PtrOf`)
def TPTR_PtrType : TPtrTypeDef<"Pointer", "ptr", [
    MemRefElementTypeInterface
]> {
    let summary = "Pointer type (`::mlir::triton::PointerType`) in Triton IR type system";

    let description = [{
        Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
    }];

    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let skipDefaultBuilders = 1;
}

//
// Op Base
//
class TPTR_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPtr_Dialect, mnemonic, traits> {
}

def TPTR_PtrToMemrefOp : TPTR_Op<"ptr_to_memref", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_MemrefToPtrOp : TPTR_Op<"memref_to_ptr", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_Add : TPTR_Op<"memref_to_ptr", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

#endif // TPTR_DIALECT
