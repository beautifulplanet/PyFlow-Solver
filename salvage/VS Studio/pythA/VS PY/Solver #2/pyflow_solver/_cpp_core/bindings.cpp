#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "solver.h"

namespace py = pybind11;

PYBIND11_MODULE(_cpp_solver, m) {
    m.doc() = "C++ accelerated CFD solver for PyFlow";
    
    py::class_<pyflow::CFDSolver>(m, "CFDSolver")
        .def(py::init<int, double, double, double>())
        .def("simulation_step", &pyflow::CFDSolver::simulation_step)
        .def("solve_lid_driven_cavity", &pyflow::CFDSolver::solve_lid_driven_cavity,
             py::arg("dt"), py::arg("total_time"), py::arg("p_iterations") = 1000)
        .def_property_readonly("u", &pyflow::CFDSolver::get_u)
        .def_property_readonly("v", &pyflow::CFDSolver::get_v)
        .def_property_readonly("p", &pyflow::CFDSolver::get_p)
        .def_property_readonly("u_residuals", &pyflow::CFDSolver::get_u_residuals)
        .def_property_readonly("v_residuals", &pyflow::CFDSolver::get_v_residuals)
        .def_property_readonly("cont_residuals", &pyflow::CFDSolver::get_cont_residuals);
}
