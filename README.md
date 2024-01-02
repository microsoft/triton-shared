# triton-shared

A shared middle-layer for the Triton Compiler.

Currently the middle layer is not complete but has enough functionality to demonstrate how it can work. The general idea is that Triton IR is lowered into an MLIR core dialect to allow it to be both shared across Triton targets as well as allow back-ends to be shared with other languages.

The basic intended architecture looks like this:

[Triton IR] -> [Middle Layer] -> [HW specific IR]

The middle-layer uses MLIR's Linalg and Tensor Dialects for operations on Triton block values. Operations on Triton pointers use the Memref Dialect.

## Motivation
[This talk at the 2023 Triton Developer Conferene](https://www.youtube.com/watch?v=y2V3ucS1pfQ) gives some backgorund on the project and its goals.

## Usage
This repo doesn't build by itself and must instead by built from within a [Triton repo](https://github.com/openai/triton) where it is included as a submodule.
To add the shared middle-layer in your Triton build do `export TRITON_CODEGEN_TRITON_SHARED=1` before invoking your build. 
Once it is part of the Triton build it can be leveraged in two ways:

### 1. Stand-Alone
The middle layer can be used as a stand-alone component to convert Triton dialect to the middle layer dialects. This is intended for testing and validation purposes, but could potentially be used before sending the IR to another MLIR complier.

Stand-alone example:
```
triton-shared-opt --triton-to-linalg %file
```

### 2. Backend Component
The intended use of the Triton middle layer is to be used as a component in a Triton back-end. This can be accomplished by adding the cmake targets it produces and its headers files to that back-end. An example back-end will be published at a later date.

## Implementation details

Even though a valid triton program can perform load and store in arbitrary memory locations, the prototype only supports lowering programs that have structured memory access patterns.

### Analyses

As part of the conversion process, there are three important analyses:

1. Pointer analysis:
    + This analysis is responsible for extracting structured memory access patterns from a `triton` program during load and store; it walks the IR and visits relevant instructions to build strided memory accesses in the `memref` dialect. The analysis is still in its early stage and does not support all scenarios.

2. Use analysis:
    + After "Pointer analysis", instructions that are part of memory address calculation will no longer be necessary in a triton program because their semantics have now been captured by `memref` operations representing strided memory accesses. To aid with removing these instructions safely, we perform `Use analysis` to mark which instructions are used *only* in address calculation (called `MetaUse`) or used in *both* address calculation and data manipulation (called `MixedUse`) operations. Those that are `MixedUse` are cloned and have their users adjusted accordingly with the goal of separating out the `MetaUse` ops so that they can be safely deleted.

3. Mask analysis:
    + This analysis is responsible for handling masked loads and stores.

### Conversion strategy

We introduce the `TritonToLinalg` pass that converts the `triton` dialect to the `linalg` dialect on *tensors*. This means the resulting IR is fully compatible with `linalg` tiling and fusion transformation passes. As mentioned in the `Pointer analysis`'s description, we do however have to deal with memref instructions at the load and store boundaries and have to convert them to tensors using `bufferization.to_tensor`. Here's a simple example of what the IR looks like:

```mlir
tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>) {
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %1 = tt.splat %afloat : (!tt.ptr<bf16>) -> tensor<128x!tt.ptr<bf16>>
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
  %afm = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xbf16>
  %3 = "tt.reduce"(%afm) ({
  ^bb0(%arg5: bf16, %arg6: bf16):
    %21 = arith.addf %arg5, %arg6 : bf16
    tt.reduce.return %21 : bf16
  }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
  tt.store %res, %3 : bf16
  tt.return
}
```

after conversion:

```mlir
func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] :
        memref<*xbf16> to memref<128xbf16, strided<[1]>>
    %alloc = memref.alloc() : memref<128xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<128xbf16, strided<[1]>> to memref<128xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
    %1 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %1[] : tensor<f32>
    %reduced = linalg.reduce ins(%0 : tensor<128xbf16>) outs(%inserted : tensor<f32>) dimensions = [0]
      (%in: bf16, %init: f32) {
        %3 = arith.extf %in : bf16 to f32
        %4 = arith.addf %3, %init : f32
        linalg.yield %4 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %2 = arith.truncf %extracted : f32 to bf16
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] :
        memref<*xbf16> to memref<1xbf16, strided<[1]>>
    affine.store %2, %reinterpret_cast_0[0] : memref<1xbf16, strided<[1]>>
    return

}

```

Important details to note:

+ `tt.load` (together with all of its related address calculation instructions such as `tt.addptr` and `tt.splat`) are lowered to a combination of `memref.reinterpret_cast`, `memref.alloc`, and `memref.copy`. After the initialization of the local buffer, we convert the memref back to a tensor using `bufferization.to_tensor`; this op is automatically removed during bufferization.

+ `tt.store` lowers to a combination of `memref.reinterpret_cast` and either `affine.store` or `memref.tensor_store`:

```
%reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [...] memref<*xf32> to memref<1024xf32>
%extracted_slice = tensor.extract_slice %15[0] [%21] [1] : tensor<1024xf32> to tensor<?xf32>
%subview = memref.subview %reinterpret_cast[0] [%21] [1] : memref<1024xf32> to memref<?xf32>
bufferization.materialize_in_destination %extracted_slice in writable %subview
```

+ element-wise `arith` and `math` operators are converted to their corresponding `linalg.generic` version.
+ `tt.dot` becomes `linalg.matmul`.
+ `tt.reduce` becomes `linalg.reduce`; known limitation: only support `addf` and `maxf` reduction in the reduction body for now.

### Testing

The prototype was tested on the following triton kernel examples:

1. vector addition
2. fused softmax
3. matrix multiplication
4. layer normalization
5. fused attention

In addition to testing on the tutorial kernels, there are many lit tests covering various scenarios.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
