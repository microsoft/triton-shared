#include <pybind11/pybind11.h>

namespace py = pybind11;

// The CPU backend with triton_shared doesn't do compilation from within python
// but rather externally through triton-shared-opt, so we leave this function
// blank.
void init_triton_triton_shared(py::module &&m) {}
