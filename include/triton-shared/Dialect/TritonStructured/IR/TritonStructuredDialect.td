#ifndef TRITON_STRUCTURED_DIALECT
#define TRITON_STRUCTURED_DIALECT

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure

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


  // base: base pointer used to contruct the tensor of pointers
  // sizes: size of tensor of pointers
  // strides: address increment from one element to the next in the
  //          corresponding dimension
  // parent_sizes: used to represent wrap-around behavior; constant zero
  //               indicate no wrap-around in the corresponding dimension
  let arguments = (ins TT_Ptr:$base,
                       DenseI64ArrayAttr:$sizes,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$parent_sizes,
                       DenseI64ArrayAttr:$static_strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_parent_sizes);

  let results = (outs TT_PtrTensor:$result);

  let assemblyFormat = [{
    $base `to` `sizes` `` `:` $sizes
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    `` `,` `offsets` `` `:`
    custom<DynamicIndexList>($offsets, $static_offsets)
    `` `,` `parent_sizes` `` `:`
    custom<DynamicIndexList>($parent_sizes, $static_parent_sizes)
    attr-dict `:` type($base) `to` type($result)
  }];


  let builders = [
    // Build with mixed static and dynamic entries.
    OpBuilder<(ins
      "Value":$base,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$parent_sizes)>,
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
    SmallVector<OpFoldResult> getMixedParentSizes() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticParentSizes(), getParentSizes(), b);
    }
  }];

  // TODO
  //let hasVerifier = 1;
  //let hasCanonicalizer = 1;
}

def TTS_LoadOp : TTS_Op<"load", [
  MemoryEffects<[MemRead]>,
  AttrSizedOperandSegments
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrTensor:$ptr,
                       Variadic<Index>:$dims,
                       DenseI64ArrayAttr:$static_dims,
                       Optional<AnyTypeOf<[TT_Float, TT_Int, TT_Ptr]>>:$other);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "ArrayRef<OpFoldResult>":$dims, "Value":$other)>,
  ];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

def TTS_StoreOp : TTS_Op<"store", [
  MemoryEffects<[MemWrite]>
]> {
  let summary = "optionally load data from in memory to fill a portion of the tensor";

  let arguments = (ins TT_PtrTensor:$ptr,
                       TT_Tensor:$value,
                       Variadic<Index>:$dims,
                       DenseI64ArrayAttr:$static_dims);

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<OpFoldResult>":$dims)>,
  ];

  // TODO
  //let hasCustomAssemblyFormat = 1;
  //let hasVerifier = 1;
}

#endif // TRITON_STRUCTURED_DIALECT
