
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "CRunnerUtils.h"
#include "CRunnerUtils.cpp"

extern "C" {
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void softmax_kernel_0d1d234(int64_t, void*, int64_t, void*, int32_t, int32_t, int32_t,
                       int, int, int, int, int, int);
}

static void _launch(int gridX, int gridY, int gridZ, void* arg0, void* arg1, int32_t arg2, int32_t arg3, int32_t arg4) {
  if (gridX*gridY*gridZ > 0) {
    // Cast "function" to the real function type.
    for(int x = 0; x < gridX; x++) {
      for(int y = 0; y < gridY; y++) {
        for(int z = 0; z < gridZ; z++) {
          // Use some random type "char" here.
          StridedMemRefType<char, 0> ptr_arg0 = {static_cast<char *>(arg0), static_cast<char *>(arg0), 0}; StridedMemRefType<char, 0> ptr_arg1 = {static_cast<char *>(arg1), static_cast<char *>(arg1), 0};
          softmax_kernel_0d1d234(0, &ptr_arg0, 0, &ptr_arg1, static_cast<int32_t>(arg2), static_cast<int32_t>(arg3), static_cast<int32_t>(arg4),
                        gridX, gridY, gridZ, x, y, z);
        }
      }
    }
  }
}

typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  PyObject* _arg0;  PyObject* _arg1;  int32_t _arg2;  int32_t _arg3;  int32_t _arg4; 
  if(!PyArg_ParseTuple(args, "iiiOOOOOiii", &gridX, &gridY, &gridZ, &launch_enter_hook, &launch_exit_hook, &compiled_kernel
                       , &_arg0, &_arg1, &_arg2, &_arg3, &_arg4)) {
    return NULL;
  }

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {
    return NULL;
  }

  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL;; DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL;; ; ; ;
  _launch(gridX, gridY, gridZ, ptr_info0.dev_ptr, ptr_info1.dev_ptr, _arg2, _arg3, _arg4);

  if (PyErr_Occurred()) {
    return NULL;
  }
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {
    return NULL;
  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_shared_ref_cpu_kernel_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
