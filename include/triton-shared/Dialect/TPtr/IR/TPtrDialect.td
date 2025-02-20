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

  let extraClassDeclaration = [{
    void registerTypes();
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
def TPTR_HelloType : TPtrTypeDef<"Pointer", "ptr", [
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

def TPTR_FromMemrefOp : TPTR_Op<"from_memref", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_ToMemrefOp : TPTR_Op<"to_memref", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_PtrAddOp : TPTR_Op<"ptradd", [
    Pure  ]> {
  let arguments = (ins AnyType:$arg, AnyType:$offset);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg $offset attr-dict `:` type($arg) `,` type($offset) `to` type($res)";
}

def TPTR_LoadOp : TPTR_Op<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load operation";
  let description = [{
    The `load` operation is used to read from memory. A load may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic load only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile load of a float variable.
    %0 = ptr.load volatile %ptr : !ptr.ptr -> f32

    // A nontemporal load of a float variable.
    %0 = ptr.load %ptr {nontemporal} : !ptr.ptr -> f32

    // An atomic load of an integer variable.
    %0 = ptr.load %ptr atomic monotonic {alignment = 8 : i64}
        : !ptr.ptr -> i64
    ```
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
    $addr
    attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def TTPTR_StoreOp : TPTR_Op<"store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store operation";
  let description = [{
    The `store` operation is used to write to memory. A store may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic store only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile store of a float variable.
    ptr.store volatile %val, %ptr : f32, !ptr.ptr

    // A nontemporal store of a float variable.
    ptr.store %val, %ptr {nontemporal} : f32, !ptr.ptr

    // An atomic store of an integer variable.
    ptr.store %val, %ptr atomic monotonic {alignment = 8 : i64}
        : i64, !ptr.ptr
    ```
  }];
  let arguments = (ins AnyType:$value,
                       AnyType:$addr);
  let assemblyFormat = [{
    $value `,` $addr
    attr-dict `:` type($value) `,` qualified(type($addr))
  }];
}


#endif // TPTR_DIALECT
