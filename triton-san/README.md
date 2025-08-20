# TritonSan
TritonSan is a dynamic analysis tool that is capable of accurately detecting bugs (e.g., buffer overflow, data race) in Triton programs. 

TritonSan leverages triton-shared, a shared middleware layer for the Triton compiler that includes a reference CPU backend, to compile Triton kernels into CPU executables. During this compilation process, TritonSan enables instrumentions for LLVM sanitizers and applies the necessary transformations to ensure they receive complete debug information. When the Triton kernel executes, it runs alongside the specified LLVM sanitizer, enabling accurate detection of bugs within the kernel.

## Table of Contents
1. [Installation](#install-tritonsan)
2. [Usage](#usage)
3. [Example](#example)
4. [Known Issues](#known-issues)
5. [How TritonSan works](#an-overview-of-the-tritonsan-workflow)

## Install TritonSan

### Supported Platforms
- Ubuntu 24.04

### Use the provided build script
Because the prebuilt LLVM downloaded by Triton’s `setup.py` excludes LLVM sanitizers (i.e., the compiler-rt subproject is disabled), we need to utilize a custom LLVM build that matches the top commit hash specified in `triton/cmake/llvm-hash.txt`.

To simplify installation, we provide a `build.sh` script that automates the entire process, including:
- setting up all prerequisites in a Python virtual environment, 
- building a custom LLVM with all required subprojects enabled (e.g., compiler-rt, openmp),
- installing triton-shared with TritonSan enabled into the `triton` submodule,
- generating the TritonSan driver script (`triton-san`).

**Note: to avoid naming conflicts, please clone `triton-shared` into an empty directory. The `build.sh` script will generate the necessary dependency folders alongside the repository.**
```sh
# ${PWD} should be an empty directory

# install required system dependencies
sudo apt-get install -y ccache clang lld

# check out triton-shared with all submodules
git clone --recurse-submodules https://github.com/microsoft/triton-shared.git

# call build.sh
triton-shared/triton-san/build.sh
```
**Note: because LLVM takes a long time to build, the initial run of `build.sh` may exceed 20 minutes. On later runs, the build script reuses the Python virtual environment and LLVM binary to speed up the build of `triton`, `triton-shared`, and `triton-san`.**
After installation, `build.sh` generates the following folders alongside the `triton-shared` repository.
```
llvm  triton-san  triton-shared  venv
```
- `llvm`: the custom LLVM source and binary,
- `venv`: Python environment with TritonSan-enabled Triton package installed,
- `triton-san`: TritonSan driver script and associated files.

## Usage
To use TritonSan, run the TritonSan driver script (`triton-san/triton-san`) with the target Triton program and its corresponding inputs. We also provide two sample Triton programs containing known bugs, which `build.sh` installs into the `triton-san/examples` directory.

TritonSan currently supports two types of LLVM sanitizers for bug detection, [AddressSanitizer (asan)](https://clang.llvm.org/docs/AddressSanitizer.html) and [ThreadSanitizer (tsan)](https://clang.llvm.org/docs/ThreadSanitizer.html). Users can select the appropriate sanitizer based on the type of bugs they want to detect.

```sh
# Run triton-san/triton-san to view usage instructions.

Usage: triton-san <sanitizer type> <original command used to launch the triton program...>.
<sanitizer type>:
  "asan": to detect buffer overflows
  "tsan": to detect data races

Example: triton-san asan python ./my_triton_program.py
```

**Note: before running TritonSan, please add the following import to the Triton program to specify the use of the CPU backend, which ensures all Triton kernels run on the CPU. The sanitizers require CPU backend in order to work.**

```python
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
```

## Example
Run `triton-san/examples/data-race.py` with TritonSan.
```sh
   triton-san/triton-san tsan python triton-san/examples/data-race.py
```

The output should resemble the figure below. In **lines 5 and 13**, the bug report identifies two memory accesses in the Triton program that contribute to the data race.
```sh
1   ==================
2   WARNING: ThreadSanitizer: data race (pid=1073103)
3   Write of size 8 at 0x7260004bd800 by thread T35:
4     #0 __tsan_memcpy ./llvm/llvm-project/compiler-rt/lib/tsan/rtl/tsan_interceptors_memintrinsics.cpp:27:3 (libclang_rt.tsan.so+0x6d24e) (BuildId: 8abada4307c45d3e9e44a078e3d55102ca9a1dc8)
5     #1 kernel ./triton-san/example/data-race.py:28:35 (__triton_shared_ref_cpu_kernel_launcher.so+0xcdaa)
6     #2 _launch(int, int, int, void*, int, _object*) (.omp_outlined_debug__) /tmp/tmplqnj6lsm/main.cxx:26:11 (__triton_shared_ref_cpu_kernel_launcher.so+0x7bc9)
7     #3 _launch(int, int, int, void*, int, _object*) (.omp_outlined) /tmp/tmplqnj6lsm/main.cxx:20:5 (__triton_shared_ref_cpu_kernel_launcher.so+0x7ca1)
8     #4 __kmp_invoke_microtask <null> (libomp.so+0xc3d98) (BuildId: fabd731ada4172bff4255cc39ed59517c481b7aa)
9     #5 _launch(int, int, int, void*, int, _object*) /tmp/tmplqnj6lsm/main.cxx:20:5 (__triton_shared_ref_cpu_kernel_launcher.so+0x764a)
10
11  Previous write of size 8 at 0x7260004bd800 by main thread:
12    #0 __tsan_memcpy ./llvm/llvm-project/compiler-rt/lib/tsan/rtl/tsan_interceptors_memintrinsics.cpp:27:3 (libclang_rt.tsan.so+0x6d24e) (BuildId: 8abada4307c45d3e9e44a078e3d55102ca9a1dc8)
13    #1 kernel ./triton-san/example/data-race.py:28:35 (__triton_shared_ref_cpu_kernel_launcher.so+0xcdaa)
14    #2 _launch(int, int, int, void*, int, _object*) (.omp_outlined_debug__) /tmp/tmplqnj6lsm/main.cxx:26:11 (__triton_shared_ref_cpu_kernel_launcher.so+0x7bc9)
15    #3 _launch(int, int, int, void*, int, _object*) (.omp_outlined) /tmp/tmplqnj6lsm/main.cxx:20:5 (__triton_shared_ref_cpu_kernel_launcher.so+0x7ca1)
16    #4 __kmp_invoke_microtask <null> (libomp.so+0xc3d98) (BuildId: fabd731ada4172bff4255cc39ed59517c481b7aa)
17    #5 _launch(int, int, int, void*, int, _object*) /tmp/tmplqnj6lsm/main.cxx:20:5 (__triton_shared_ref_cpu_kernel_launcher.so+0x764a)
18    #6 <null> <null> (python3.12+0x581d6f) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
19
20  Location is heap block of size 1024 at 0x7260004bd800 allocated by main thread:
21    #0 posix_memalign ./llvm/llvm-project/compiler-rt/lib/tsan/rtl/tsan_interceptors_posix.cpp:882:3 (libclang_rt.tsan.so+0x713f9) (BuildId: 8abada4307c45d3e9e44a078e3d55102ca9a1dc8)
22    #1 c10::alloc_cpu(unsigned long) <null> (libc10.so+0x808b0) (BuildId: d3883d21ef7bef12d89784e7cf1f2ab7d942cc1d)
23
24  Thread T35 (tid=1073161, running) created by main thread at:
25    #0 pthread_create ./llvm/llvm-project/compiler-rt/lib/tsan/rtl/tsan_interceptors_posix.cpp:1051:3 (libclang_rt.tsan.so+0x71aff) (BuildId: 8abada4307c45d3e9e44a078e3d55102ca9a1dc8)
26    #1 __kmp_create_worker <null> (libomp.so+0xa3055) (BuildId: fabd731ada4172bff4255cc39ed59517c481b7aa)
27
28  SUMMARY: ThreadSanitizer: data race ./triton-san/example/data-race.py:28:35 in kernel
29  ==================
```

Similarly, running `triton-san/triton-san asan python triton-san/examples/buffer-overflow.py` should produce the following output. Line 5 identifies the root cause of the buffer overflow in the Triton program.
```sh
1   =================================================================
2   ==1074677==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7c4fdf343104 at pc 0x7dffde70ccd5 bp 0x7fff04555ca0 sp 0x7fff04555460
3   WRITE of size 8192 at 0x7c4fdf343104 thread T0
4     #0 0x7dffde70ccd4 in __asan_memcpy ./llvm/llvm-project/compiler-rt/lib/asan/asan_interceptors_memintrinsics.cpp:63:3
5     #1 0x7dffdd92cce1 in kernel ./triton-san/example/buffer-overflow.py:28:35
6     #2 0x7dffdd928626 in _launch(int, int, int, void*, int, _object*) /tmp/tmp7zn051vm/main.cxx:26:11
7     #3 0x7dffdd927ff8 in launch(_object*, _object*) /tmp/tmp7zn051vm/main.cxx:106:3
8     #4 0x000000581d6f  (/usr/bin/python3.12+0x581d6f) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
9     #5 0x00000054b13b in PyObject_Call (/usr/bin/python3.12+0x54b13b) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
10    #6 0x0000005dad15 in _PyEval_EvalFrameDefault (/usr/bin/python3.12+0x5dad15) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
11    #7 0x00000054a7d1 in _PyObject_Call_Prepend (/usr/bin/python3.12+0x54a7d1) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
12    #8 0x0000005a3147  (/usr/bin/python3.12+0x5a3147) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
13    #9 0x00000054b13b in PyObject_Call (/usr/bin/python3.12+0x54b13b) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
14    #10 0x0000005dad15 in _PyEval_EvalFrameDefault (/usr/bin/python3.12+0x5dad15) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
15    #11 0x00000054cb93  (/usr/bin/python3.12+0x54cb93) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
16    #12 0x00000054b1b8 in PyObject_Call (/usr/bin/python3.12+0x54b1b8) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
17    #13 0x0000005dad15 in _PyEval_EvalFrameDefault (/usr/bin/python3.12+0x5dad15) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
18    #14 0x0000005d500a in PyEval_EvalCode (/usr/bin/python3.12+0x5d500a) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
19    #15 0x0000006081e1  (/usr/bin/python3.12+0x6081e1) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
20    #16 0x0000006b5032  (/usr/bin/python3.12+0x6b5032) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
21    #17 0x0000006b4d99 in _PyRun_SimpleFileObject (/usr/bin/python3.12+0x6b4d99) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
22    #18 0x0000006b4bce in _PyRun_AnyFileObject (/usr/bin/python3.12+0x6b4bce) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
23    #19 0x0000006bcc34 in Py_RunMain (/usr/bin/python3.12+0x6bcc34) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
24    #20 0x0000006bc71c in Py_BytesMain (/usr/bin/python3.12+0x6bc71c) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
25    #21 0x7dffde22a1c9 in __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16
26    #22 0x7dffde22a28a in __libc_start_main csu/../csu/libc-start.c:360:3
27    #23 0x0000006575a4 in _start (/usr/bin/python3.12+0x6575a4) (BuildId: 7b4e3ff9cbc7f8717dfff5daeb5d187eee8b2088)
28
29  0x7c4fdf343104 is located 0 bytes after 8196-byte region [0x7c4fdf341100,0x7c4fdf343104)
allocated by thread T0 here:
30    #0 0x7dffde70fa17 in posix_memalign ./llvm/llvm-project/compiler-rt/lib/asan/asan_malloc_linux.cpp:139:3
31    #1 0x79ffdad6d8b0 in c10::alloc_cpu(unsigned long) (./venv/lib/python3.12/site-packages/torch/lib/libc10.so+0x808b0) (BuildId: d3883d21ef7bef12d89784e7cf1f2ab7d942cc1d)
32
33  SUMMARY: AddressSanitizer: heap-buffer-overflow ./triton-san/example/buffer-overflow.py:28:35 in kernel
```

In both examples, TritonSan’s output should correspond to the bug description provided at the beginning of each sample program. 

**We also found that these two bugs are not detected by [Compute Sanitizer](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-third-party-tools), highlighting the value of TritonSan as a complementary debugging tool for Triton programs.**

## Known Issues
### Warning generated from LLVM sanitizers
When using 'tsan' for data race detection, LLVM sanitizers may emit the following warning. This message is harmless and can be safely ignored for now.

```sh
Warning: please export TSAN_OPTIONS='ignore_noninstrumented_modules=1' to avoid false positive reports from the OpenMP runtime!
```

## An Overview of the TritonSan Workflow
![A figure illustrating TritonSan's workflow. TritonSan uses the LD_PRELOAD environment variable to initialize the selected LLVM sanitizer before launching the Triton program. For each encountered Triton kernel, it applies an instrumentation pass on the generated LLVM IR and links the result with the sanitizer's runtime library. Finally, TritonSan executes the Triton kernel on the CPU to perform dynamic analysis and detect potential bugs.](image/workflow.png)