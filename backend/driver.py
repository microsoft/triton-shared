import hashlib
import tempfile
import sysconfig

import os, subprocess, tempfile
import importlib.util
import sysconfig

from pathlib import Path

from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
library_dir = [os.path.join(dirname, "lib")]
libraries = ['amdhip64']

def compile_module_from_src(src, name):
    key = hashlib.md5(src.encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir, library_dir, include_dir, libraries)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, cache_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def _extracted_ty(ty):
    if ty[0] == '*':
        return "PyObject*"
    return {
        'i1': 'int32_t',
        'i32': 'int32_t',
        'i64': 'int64_t',
        'u32': 'uint32_t',
        'u64': 'uint64_t',
        'fp16': 'float',
        'bf16': 'float',
        'fp32': 'float',
        'f32': 'float',
        'fp64': 'double',
    }[ty]

def _format_of(ty):
    return {
        "PyObject*": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "uint32_t": "I",
        "int32_t": "i",
        "uint64_t": "K",
        "int64_t": "L",
    }[ty]

def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    format = "iiiOOO" + ''.join([_format_of(_extracted_ty(ty)) for ty in signature.values()])
    return f"""
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "CRunnerUtils.h"
#include "CRunnerUtils.cpp"

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if i not in constants)},
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    // Cast "function" to the real function type.
    for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          // Use some random type "char" here.
          {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}
          {kernel_name}({', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}" for i, ty in signature.items() if i not in constants)},
                        gridX, gridY, gridZ, x, y, z);
        }}
      }}
    }}
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_ty(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &launch_enter_hook, &launch_exit_hook, &compiled_kernel
                       {', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {{
    return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {{
    return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""

def get_kernel_bin():
    return "cpuasm"


def make_launcher(launcher_src, kernel_placeholder_name):
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    def launch(
        gridX, gridY, gridZ, num_warps, num_ctas, clusterDim0, clusterDim1, clusterDim2,
        shared, stream, cu_function, launch_enter_hook, launch_exit_hook, compiled_kernel,
        *args):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile a kernel every time.
        asm_src = compiled_kernel.asm["cpuasm"]
        src = launcher_src.replace(kernel_placeholder_name, compiled_kernel.metadata["name"])
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_src_path = os.path.join(tmpdir, "kernel.s")
            launcher_src_path = os.path.join(tmpdir, "main.cxx")
            so_path = os.path.join(tmpdir, "kernel.so")
            Path(asm_src_path).write_text(asm_src)
            Path(launcher_src_path).write_text(src)
            # Compile it together.
            subprocess.check_call(["g++", launcher_src_path, asm_src_path, f"-I{py_include_dir}", f"-I{Path(__file__).resolve().parent}", "-shared", "-fPIC", "-o", so_path])

            # Load and launch the compiled kernel.
            spec = importlib.util.spec_from_file_location("__triton_shared_ref_cpu_kernel_launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, compiled_kernel, *args)

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        ids = {
            "ids_of_folded_args": metadata.get("ids_of_folded_args", tuple()), "ids_of_const_exprs":
            src.fn.constexprs if hasattr(src, "fn") else tuple()
        }
        constants = src.constants if hasattr(src, "constants") else dict()

        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"
        launcher_src = _generate_launcher(constants, src.signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = make_launcher(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        self.launch(*args, **kwargs)


class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.launcher_cls = CPULauncher

    @staticmethod
    def is_active():
        import torch
        return torch.version.hip is not None

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self):
        return None

    def get_current_device(self):
        # CPU doesn't have a device to return.  Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        device = self.get_current_device()
        arch = self.utils.get_device_properties(device)['arch']
        return ("hip", arch.split(':')[0])

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
