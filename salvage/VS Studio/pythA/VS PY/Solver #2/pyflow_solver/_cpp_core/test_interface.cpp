#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Simple function to modify a NumPy array in-place
void force_interior_value(py::array_t<double, py::array::c_style | py::array::forcecast> arr, double value_to_set) {
    // Request mutable buffer information
    py::buffer_info buf = arr.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }

    auto rows = buf.shape[0];
    auto cols = buf.shape[1];
    double *ptr = static_cast<double *>(buf.ptr);

    // Loop over the *interior* of the array
    for (ssize_t i = 1; i < rows - 1; i++) {
        for (ssize_t j = 1; j < cols - 1; j++) {
            ptr[i * cols + j] = value_to_set;
        }
    }
}

PYBIND11_MODULE(_cpp_test_interface, m) {
    m.doc() = "Test module for verifying Python/C++ array passing behavior"; 
    m.def("force_interior_value", &force_interior_value, 
          "Test function that modifies a NumPy array in-place by setting interior values");
}
