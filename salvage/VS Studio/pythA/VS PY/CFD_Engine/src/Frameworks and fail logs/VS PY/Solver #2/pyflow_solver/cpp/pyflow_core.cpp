#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// A simple function that takes two NumPy arrays and returns their sum.
// pybind11 automatically handles the conversion between np.ndarray and py::array_t.
py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    // Request buffer information from the input arrays
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
    if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1])
        throw std::runtime_error("Input shapes must match");

    // Create a new NumPy array to store the result
    auto result = py::array_t<double>(buf1.shape);
    py::buffer_info buf_res = result.request();

    // Get pointers to the data
    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr_res = static_cast<double *>(buf_res.ptr);

    // Perform the addition (this is the part that runs at C++ speed)
    for (size_t i = 0; i < buf1.size; i++) {
        ptr_res[i] = ptr1[i] + ptr2[i];
    }

    return result;
}

// The PYBIND11_MODULE macro creates the entry point for the Python interpreter
PYBIND11_MODULE(pyflow_core, m) {
    m.doc() = "High-performance core for PyFlow solver"; // Optional module docstring
    m.def("add_arrays", &add_arrays, "A function that adds two NumPy arrays");
}
