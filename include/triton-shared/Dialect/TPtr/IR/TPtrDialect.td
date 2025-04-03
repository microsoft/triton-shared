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
    "mlir::ptr::PtrDialect"
  ];

  let usePropertiesForAttributes = 1;
}

class TPtrTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TPtr_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

//
// Op Base
//
class TPTR_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPtr_Dialect, mnemonic, traits> {
}

def TPTR_IntToPtrOp : TPTR_Op<"inttoptr", [
    Pure
  ]> {
  let summary = "Integer to a pointer operation";
  let description = [{
    The `inttoptr` operation casts an int or index value to a pointer.

    Example:
    ```mlir
    %ptr = ptr.inttoptr %int : i32 to !ptr.ptr<1 : i32>
    ```
  }];
  let arguments = (ins AnySignlessIntegerOrIndex:$arg);
  let results = (outs Ptr_PtrType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_PtrToIntOp : TPTR_Op<"ptrtoint", [
    Pure
  ]> {
  let summary = "Pointer to an integer operation";
  let description = [{
    The `ptrtoint` operation casts a pointer value to an int or index.

    Example:
    ```mlir
    %int = ptr.ptrtoint %ptr : !ptr.ptr<1 : i32> to i32
    ```
  }];
  let arguments = (ins Ptr_PtrType:$arg);
  let results = (outs AnySignlessIntegerOrIndex:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_TypeOffsetOp : TPTR_Op<"type_offset", [ConstantLike, Pure]> {
  let summary = "Creates a type offset constant.";
  let description = [{
    The `addr.type_offset` operation produces an int or index-typed SSA value
    equal to a target-specific constant representing the offset of a single
    element of the given type. The default return type is `index`.
    Example:

    ```mlir
    %0 = addr.type_offset f32
    %1 = addr.type_offset memref<12 x f64> : i32
    ```
  }];

  let arguments = (ins TypeAttr:$baseType);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let builders = [
    OpBuilder<(ins "TypeAttr":$baseType, CArg<"Type", "nullptr">:$resultTy)>
  ];
  let assemblyFormat = [{
     attr-dict $baseType custom<IntType>(type($result))
  }];
  let hasFolder = 1;
}

def TPTR_FromMemrefOp : TPTR_Op<"from_memref", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def TPTR_ToMemrefOp : TPTR_Op<"to_memref", [
    Pure  ]> {
  let arguments = (ins Ptr_PtrType:$arg);
  let results = (outs AnyStaticShapeMemRef:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPTR_PtrAddOp : TPTR_Op<"ptradd", [Pure, AllTypesMatch<["base", "result"]>]> {
  let summary = "Pointer-index add operation";
  let description = [{
    The `ptradd` operation adds an `address` and an integer or index to
    produce a new address.

    Example:
    ```mlir
    %addr = ptr.ptradd %addr : !ptr.ptr<3 : i32>, %c10 : i32
    ```
  }];

  let arguments = (ins Ptr_PtrType:$base, AnySignlessIntegerOrIndex:$offset);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = "$base $offset attr-dict `:` type($base) `,` type($offset) `to` type($result)";
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
