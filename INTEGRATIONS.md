# Integrating with triton-shared

There are many ways to integrate `triton-shared` with your existing `triton` compiler. This section describes a suggested approach that Microsoft uses internally for the maia compiler backend.

## 1. Project structure

We assume that you already have a fork of the official `triton` repository.

## 2. Your copy of `triton-shared`

You can either create a `git submodule` or a `git subtree` of `triton-shared` for your project.
Use `git submodule` if you don't intend to make custom changes to `triton-shared`, otherwise, `git subtree` is a better fit.

We do not have any requirements on where the `triton-shared` folder should live as this is configurable in the CMake script. Some suggestions are:

If your project is a mono repo where the triton fork lives together with other projects, we recommend adding `triton-shared` to the root of your mono repo:

```
--- your compiler backend repository
                                   |---- triton
                                   |---- other projects
                                   |---- ...
                                   |---- triton-shared
```


If your repository only contains the `triton` fork, you can place `triton-shared` under the root `triton` directory:


```
--- your triton fork
                   |---- triton-shared
                   |---- libs
                   |---- ...

```

## 3. Add `triton-shared`'s include directories

Next, we need to add `triton-shared` include directories to the `triton` builds. This allows your code to include `triton-shared` analyses and passes through:

```
...
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
...
```

In your `triton/CMakeLists.txt` file, there is a section that specifies all the include directories when building `triton` (https://github.com/triton-lang/triton/blob/16e6390b5ee79f31fa8062d441fd203a34c8afbd/CMakeLists.txt#L112). Add the `triton-shared` include directory there:

```cmake
...

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) # Tablegen'd files

# Set `TRITON_SHARED_INCLUDE_DIR` to point to `triton-shared/include`
# The path here depends on where you place triton-shared relative to
# your triton repo.
include_directories(${TRITON_SHARED_INCLUDE_DIR})
```

## 4. Build `triton-shared` with `triton`

`triton-shared` comes with an optional sample CPU backend for `triton`. We support both building with and without the CPU backend.

To prepare, first, set `TRITON_SHARED_SRC_ROOT` to point to the root directory of `triton-shared`.

To build with the CPU backend, right after the call to `include_directories(${TRITON_SHARED_INCLUDE_DIR})` in step 4., put the following:

```cmake
set(TRITON_PLUGIN_DIRS ${TRITON_SHARED_SRC_ROOT})
```

To build *without* the CPU backend, right after the `triton-shared` include call, put the following:

```cmake
set(TRITON_SHARED_BUILD_CPU_BACKEND OFF)
add_subdirectory(${TRITON_SHARED_SRC_ROOT})
```

## 5. Test if `triton-shared` was built correctly

Run `ninja check-triton-shared-lit-tests` under `triton`'s root build folder to check if the `triton-shared` lit tests are passing.

## 6. Issues

Please submit any build related issues to our GitHub repository with instructions on how to reproduce the errors. We appreciate any feedback and bug reports!
