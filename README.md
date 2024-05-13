# triton-shared

A shared middle-layer for the Triton Compiler.

Currently the middle layer is not complete but has enough functionality to demonstrate how it can work. The general idea is that Triton IR is lowered into an MLIR core dialect to allow it to be both shared across Triton targets as well as allow back-ends to be shared with other languages.

The basic intended architecture looks like this:

```
[Triton IR] -> [Middle Layer] -> [HW specific IR]
```

The middle-layer uses MLIR's Linalg and Tensor Dialects for operations on Triton block values. Operations on Triton pointers use the Memref Dialect.

## Motivation

[This talk at the 2023 Triton Developer Conferene](https://www.youtube.com/watch?v=y2V3ucS1pfQ) gives some background on the project and its goals.

## Usage

This section describes how to build `triton-shared` as a standalone project. For integration with existing `triton` compiler, please refer to the [integrations doc](INTEGRATIONS.md).

This repo now includes `triton` as a submodule and builds as an out-of-tree backend.

To build this repo, clone `triton-shared` to a folder called `triton_shared` (notice the **underscore**).
`Triton` will use this folder name to create a module under `triton.runtime` for the reference CPU backend.

You need to set the `TRITON_PLUGINS_DIRS` environment variable to the location of your `triton-shared` directory for `triton` to find it.

```
export TRITON_PLUGIN_DIRS=$(pwd)/triton_shared

git clone --recurse-submodules https://github.com/microsoft/triton-shared.git triton_shared
cd triton_shared/triton/python
```

To build with Clang:

```sh
python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist
sudo apt-get update -y
sudo apt-get install -y ccache clang lld
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv '.[tests]'
```

To build with a virtualenv:

```
python3 -m venv .venv --prompt triton
source .venv/bin/activate

pip3 install ninja cmake wheel pytest
pip3 install -e python --no-build-isolation
```

The resulting `triton-shared` binaries will be placed under `triton/python/build/{current_cmake_version}/third_party/triton_shared`

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

## Other resources

[Internals](INTERNALS.md) of `triton-shared` which discusses our lowering strategy from `triton` IR.

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
