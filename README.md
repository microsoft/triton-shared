**This repository is no longer maintained.**
It remains available for reference.
For continued development, please consider forking the repository.

# triton-shared

A shared middle-layer for the Triton Compiler.

Currently the middle layer is not complete but has enough functionality to demonstrate how it can work. The general idea is that Triton IR is lowered into an MLIR core dialect to allow it to be both shared across Triton targets as well as allow back-ends to be shared with other languages.

The basic intended architecture looks like this:

[Triton IR] -> [Middle Layer] -> [HW specific IR]

The middle-layer uses MLIR's Linalg and Tensor Dialects for operations on Triton block values. Operations on Triton pointers use the Memref Dialect.

## Motivation
[This talk at the 2023 Triton Developer Conferene](https://www.youtube.com/watch?v=y2V3ucS1pfQ) gives some background on the project and its goals.

## Usage

This repo now includes `triton` as a submodule and builds as an out-of-tree backend.

To build this repo clone `triton-shared` to a folder called `triton_shared` (notice the **underscore**).
`Triton` will use this folder name to create a module under `triton.runtime` for the reference CPU backend.

You need to set the `TRITON_PLUGIN_DIRS` environment variable to the location of your `triton-shared` directory for `triton` to find it.

```
export TRITON_PLUGIN_DIRS=$(pwd)/triton_shared

git clone https://github.com/microsoft/triton-shared.git triton_shared
git clone https://github.com/triton-lang/triton.git
cd triton && git checkout $(cat ../triton_shared/triton-hash.txt)
```

To build with Clang:

```sh
python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist pybind11 setuptools
sudo apt-get update -y
sudo apt-get install -y ccache clang lld
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv '.[tests]'
```

To build with a virtualenv:

```
python3 -m venv .venv --prompt triton
source .venv/bin/activate

pip3 install ninja cmake wheel pytest pybind11 setuptools
pip3 install -e . --no-build-isolation
```

The resulting `triton-shared` binaries will be placed under `triton/build/{current_cmake_version}/third_party/triton_shared`

### 1. Stand-Alone
The middle layer can be used as a stand-alone component to convert Triton dialect to the middle layer dialects. This is intended for testing and validation purposes, but could potentially be used before sending the IR to another MLIR complier.

Stand-alone example:
```
triton-shared-opt --triton-to-linalg %file
```

### 2. Backend Component
The intended use of the Triton middle layer is to be used as a component in a Triton back-end. This can be accomplished by adding the cmake targets it produces and its headers files to that back-end. An example back-end will be published at a later date.

### 3. Reference CPU backend
We also include an experimental reference CPU backend that leverages all existing `mlir` passes. After building, the CPU backend can be used by setting `triton`'s active driver:

```python

import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

For more examples, please refer to `python/examples`.

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
  %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
  %3 = "tt.reduce"(%afm) ({
  ^bb0(%arg5: bf16, %arg6: bf16):
    %21 = arith.addf %arg5, %arg6 : bf16
    tt.reduce.return %21 : bf16
  }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
  tt.store %res, %3 : !tt.ptr<bf16>
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

1. [vector addition](./python/examples/test_vec_add.py)
2. [fused softmax](./python/examples/test_softmax.py)
3. [matrix multiplication](./python/examples/test_matmul.py)
4. layer normalization
5. fused attention

The Python tests are setup to run with Pytest and you will need to set the following environment variables to run them:
```
export LLVM_BINARY_DIR=<path-to-your-llvm-binaries>
export TRITON_SHARED_OPT_PATH=$TRITON_PLUGIN_DIRS/triton/build/<your-cmake-directory>/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

pytest <path-to-triton-shared>/python/examples
```
In addition to testing on the tutorial kernels, there are many lit tests covering various scenarios.

## Intermediate Representation (IR) Dumps

To facilitate debugging and analysis, the triton-shared project now supports emitting all intermediate representations (IRs) generated during the compilation process. This functionality is controlled via the environment variable `TRITON_SHARED_DUMP_PATH`.

### How It Works

By setting the `TRITON_SHARED_DUMP_PATH` environment variable, you specify a directory where all intermediate representations will be saved. The Triton compiler will emit IR dumps at various stages of compilation into the specified folder, allowing developers to inspect and analyze the transformations applied to the code.

### How to Use

Create a directory where the IR dumps will be stored (e.g., /path/to/dump_dir).
Set the `TRITON_SHARED_DUMP_PATH` environment variable to the directory path:
`export TRITON_SHARED_DUMP_PATH=/path/to/dump_dir`
Run your Triton compilation as usual. The compiler will emit IR dumps into the specified directory.

### Example

Suppose your dump directory is `/tmp/ir_dumps`. Before running your code, set the environment variable:

```sh
export TRITON_SHARED_DUMP_PATH=/tmp/ir_dumps
```

After the compilation process completes, you can explore the `/tmp/ir_dumps` directory to find all the intermediate representation files.

```sh
$ ls /tmp/ir_dumps
ll.ir  ll.mlir  tt.mlir  ttshared.mlir
```

## Debugging Triton Programs
Triton-shared includes a build option that enables LLVM-sanitizers - AddressSanitizer (ASan) and ThreadSanitizer (TSan) - to help detect memory safety and concurrency issues in Triton programs. These sanitizers dynamically analyze the program during execution, identifying bugs such as buffer overflows and data races respectively. For more details and setup instructions, refer [here](triton-san/README.md).

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
