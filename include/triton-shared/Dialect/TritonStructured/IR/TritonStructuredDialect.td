#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"

def Triton_Structured_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::tts";

  let summary = "Structured Triton operations";

  let description = [{
    Triton Structured Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect"
  ];

  let usePropertiesForAttributes = 1;
}

//
// Op Base
//
class TTS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Structured_Dialect, mnemonic, traits> {
}

def TTS_MakeTensorPtrOp
  : TTS_Op<"make_tptr", [AttrSizedOperandSegments, Pure]> {
  let summary = "create a pointer that points to a tensor in memory";

  // base:    Base pointer used to contruct the tensor of pointers or pointer to tensor.
  // sizes:   Size of the data being loaded or stored.
  // strides: The strides of the parent tensor, which means how much to increase the pointer
  //          by when moving by 1 element in a specific axis.
  // order:   The order of the block, which means how the block is laid out in memory.
  //          It contains the same info as order in tt.make_tensor_ptr.
  // shape:   If order is present, this field signifies the shape of the parent tensor in
  //          memory; if order is not present, it signifies the boundary by which addresses
  //          wraps around (constant zero indicates no wrap-around in the corresponding dimension).
  // offsets: Offset of the block along each dimension from base.
  // result:  If order is present, this op produces a pointer to a tensor; otherwise,
  //          it produces a tensor of pointers.

  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$shape,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_shape,
                       DenseI32ArrayAttr:$order);

  let results = (outs TT_PtrLike:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `shape` `` `:`
    custom<DynamicIndexList>($shape, $static_shape)
    `` `,` `order` `` `:` $order
    attr-dict `:` type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$shape,
      "ArrayRef<int32_t>":$order)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedSizes() {
      Builder b(getContext());
      SmallVector<Value> dynSizes; // sizes are always static
      return ::mlir::getMixedValues(getSizes(), dynSizes, b);
    }
    SmallVector<OpFoldResult> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    SmallVector<OpFoldResult> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedShape() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticShape(), getShape(), b);
    }
    bool isBlockPtr() {
      return !getOrder().empty();
    }
    bool isStructuredPtr() {
      return !isBlockPtr() &&
             llvm::all_of(getStaticShape(), [](auto shape) { return shape == 0; });
    }
    bool isSplitPtr() {
      return !isBlockPtr() &&
             !isStructuredPtr();
    }
  }];

  // TODO
  //let hasVerifier = 1;
  //let hasCanonicalizer = 1;
}

def TTS_GetStructuredStateOp : TTS_Op<"get_structured_state", [AttrSizedResultSegments, Pure]> {
  let summary = "Placeholder for the structured pointer states computed during PtrAnalysis.";
  let description = "Used to pass the offsets and strides to scf.for op to simplify IR rewrites.";

  let arguments = (ins AnyTypeOf<[TT_PtrLike, I32Tensor]>:$input);
  let results = (outs AnyTypeOf<[TT_PtrLike, I32Tensor]>:$structured, Variadic<Index>:$offsets, Variadic<Index>:$strides);

  let builders = [
    OpBuilder<(ins "Value":$input)>,
  ];

  let extraClassDeclaration = [{
    static std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
      getOffsetAndStrideTypes(MLIRContext *context, Type ptrLikeType);

    static std::optional<std::pair<int32_t, int32_t>>
      getOffsetAndStrideSegmentSizes(Type ptrLikeType);
  }];

  let hasFolder = 0;
  let hasVerifier = 1;
}

def TTS_GatherOp : TTS_Op<"gather", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<"mask type matches ptr type", "offset", "mask", "triton::getI1SameShape($_self)">,
  OptionalTypesMatchWith<"other matches ptr type", "ptr", "other", "triton::getPointeeType($_self)">
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other
  );

  let results = (outs TT_Type:$result);

  let assemblyFormat = [{
    $ptr `[` $offset `]` (`mask` `=` $mask^)? (`default` `=` $other^)?
    attr-dict `:` `(` type($ptr) `,` type($offset) `)` `->` type($result)
  }];
}

def TTS_ScatterOp : TTS_Op<"scatter", [
  MemoryEffects<[MemWrite]>,
  OptionalTypesMatchWith<"mask type matches offset type", "offset", "mask",
                 "triton::getI1SameShape($_self)">
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  let arguments = (
    ins
    TT_Ptr:$ptr,
    TT_IntLike:$offset,
    TT_Type:$value,
    Optional<TT_BoolLike>:$mask
  );

  let assemblyFormat = [{
    $value `into` $ptr `[` $offset `]` (`mask` `=` $mask^)?
    attr-dict `:` type($value) `into` ` ` `(` type($ptr) `,` type($offset) `)`
  }];
}

def TTS_LoadOp : TTS_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "ArrayRef<OpFoldResult>":$mask_dims, "Value":$other)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

def TTS_StoreOp : TTS_Op<"store", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "optionally store data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrLike:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$mask_dims,
                       DenseI64ArrayAttr:$static_mask_dims);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<OpFoldResult>":$dims)>,
  ];

  let extraClassDeclaration = [{
    /// Return a vector of all the static or dynamic fields
    SmallVector<OpFoldResult> getMixedMaskDims() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticMaskDims(), getMaskDims(), b);
    }

    bool hasMask() {
      return !getMixedMaskDims().empty();
    }
  }];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}


def TTS_PtrAddOp : TTS_Op<"ptradd", [
    Pure, AllTypesMatch<["base", "result"]>
  ]> {
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
  let assemblyFormat = [{
    $base custom<PtrType>(type($base)) `,` $offset
    custom<IntType>(type($offset)) attr-dict
  }];
}


//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def TTS_PtrLoadOp : TTS_Op<"ptr_load", [
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
  let arguments = (ins Ptr_PtrType:$addr);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
    $addr
    attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def TTS_PtrStoreOp : TTS_Op<"ptr_store", [
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
                       Ptr_PtrType:$addr);
  let assemblyFormat = [{
    $value `,` $addr
    attr-dict `:` type($value) `,` qualified(type($addr))
  }];
}

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//

def TTS_IntToPtrOp : TTS_Op<"inttoptr", [
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
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//

def TTS_PtrToIntOp : TTS_Op<"ptrtoint", [
    Pure  ]> {
  let summary = "Pointer to an integer operation";
  let description = [{
    The `ptrtoint` operation casts a pointer value to an int or index.

    Example:
    ```mlir
    %int = ptr.ptrtoint %ptr : !ptr.ptr<1 : i32> to i32
    ```
  }];
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}


#endif // TRITON_STRUCTURED_DIALECT
