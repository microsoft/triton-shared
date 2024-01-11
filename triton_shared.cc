#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "llvm/IR/Constants.h"
#include <mutex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_triton_shared(py::module &&m) {}
