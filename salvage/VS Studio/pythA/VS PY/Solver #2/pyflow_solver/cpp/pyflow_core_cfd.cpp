

#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Function to solve the pressure Poisson equation using Jacobi iteration
py::array_t<double> solve_pressure_poisson(
    py::array_t<double> b,       // RHS of the Poisson equation
    py::array_t<double> p_init,  // Initial pressure field
    double dx,                   // Grid spacing in x-direction
    double dy,                   // Grid spacing in y-direction
    int max_iter,                // Maximum iterations
    double tolerance,            // Convergence tolerance
    double alpha_p               // Pressure under-relaxation factor
) {
    // Access the input arrays
    py::buffer_info buf_b = b.request();
    py::buffer_info buf_p = p_init.request();
    
    // Check dimensions
    if (buf_b.ndim != 2 || buf_p.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");
    
    if (buf_b.shape[0] != buf_p.shape[0] || buf_b.shape[1] != buf_p.shape[1])
        throw std::runtime_error("Input arrays must have the same shape");
    
    int ny = buf_b.shape[0];
    int nx = buf_b.shape[1];
    
    // Create a copy of p_init to work with
    auto p = py::array_t<double>(buf_p.shape);
    py::buffer_info buf_p_out = p.request();
    
    // Get raw pointers to the data
    double* ptr_b = static_cast<double*>(buf_b.ptr);
    double* ptr_p_init = static_cast<double*>(buf_p.ptr);
    double* ptr_p = static_cast<double*>(buf_p_out.ptr);
    
    // Copy initial pressure field
    for (size_t i = 0; i < buf_p.size; i++) {
        ptr_p[i] = ptr_p_init[i];
    }
    
    // Calculate coefficient for the finite difference stencil
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double coef = 2.0 * (1.0/dx2 + 1.0/dy2);
    
    // Create a temporary array for Jacobi iteration
    std::vector<double> p_temp(nx * ny);
    
    // Iterate until convergence or max iterations
    double error = 1.0;
    int iter = 0;
    
    while (error > tolerance && iter < max_iter) {
        error = 0.0;
        
        // Jacobi iteration
        for (int i = 1; i < ny-1; i++) {
            for (int j = 1; j < nx-1; j++) {
                int idx = i * nx + j;
                int idx_n = (i+1) * nx + j;  // north
                int idx_s = (i-1) * nx + j;  // south
                int idx_e = i * nx + (j+1);  // east
                int idx_w = i * nx + (j-1);  // west
                
                double p_new = (
                    (ptr_p[idx_e] + ptr_p[idx_w]) / dx2 +
                    (ptr_p[idx_n] + ptr_p[idx_s]) / dy2 -
                    ptr_b[idx]
                ) / coef;
                
                // Apply under-relaxation
                p_temp[idx] = ptr_p[idx] + alpha_p * (p_new - ptr_p[idx]);
                
                // Calculate error
                error += std::abs(p_temp[idx] - ptr_p[idx]);
            }
        }
        
        // Update pressure field
        for (int i = 1; i < ny-1; i++) {
            for (int j = 1; j < nx-1; j++) {
                int idx = i * nx + j;
                ptr_p[idx] = p_temp[idx];
            }
        }
        
        // Enforce boundary conditions (zero gradient)
        // Bottom and top boundaries
        for (int j = 0; j < nx; j++) {
            ptr_p[j] = ptr_p[nx + j];              // bottom
            ptr_p[(ny-1) * nx + j] = ptr_p[(ny-2) * nx + j];  // top
        }
        
        // Left and right boundaries
        for (int i = 0; i < ny; i++) {
            ptr_p[i * nx] = ptr_p[i * nx + 1];          // left
            ptr_p[i * nx + (nx-1)] = ptr_p[i * nx + (nx-2)];  // right
        }
        
        // Calculate relative error
        error /= (nx * ny);
        iter++;
    }
    
    // Print convergence information
    std::cout << "Pressure solver converged in " << iter << " iterations with error " << error << std::endl;
    return p;
}

// Function to calculate the source term for pressure equation
py::array_t<double> calculate_pressure_source(
    py::array_t<double> u,       // x-velocity field
    py::array_t<double> v,       // y-velocity field
    double dx,                   // Grid spacing in x-direction
    double dy,                   // Grid spacing in y-direction
    double dt                    // Time step
) {
    // Access the input arrays
    py::buffer_info buf_u = u.request();
    py::buffer_info buf_v = v.request();
    
    if (buf_u.ndim != 2 || buf_v.ndim != 2)
        throw std::runtime_error("Velocity arrays must be 2D");
    
    int ny = buf_u.shape[0];
    int nx = buf_u.shape[1];
    
    if (buf_v.shape[0] != ny || buf_v.shape[1] != nx)
        throw std::runtime_error("Velocity arrays must have the same shape");
    
    // Create output array for the source term
    auto b = py::array_t<double>(py::array::ShapeContainer{ny, nx});
    py::buffer_info buf_b = b.request();
    
    // Get raw pointers
    double* ptr_u = static_cast<double*>(buf_u.ptr);
    double* ptr_v = static_cast<double*>(buf_v.ptr);
    double* ptr_b = static_cast<double*>(buf_b.ptr);
    
    // Compute the source term: -(1/dt) * div(u)
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int idx = i * nx + j;
            
            // Use central differences for the divergence
            double du_dx = (ptr_u[i * nx + (j+1)] - ptr_u[i * nx + (j-1)]) / (2.0 * dx);
            double dv_dy = (ptr_v[(i+1) * nx + j] - ptr_v[(i-1) * nx + j]) / (2.0 * dy);
            
            // Source term (negative divergence divided by dt)
            ptr_b[idx] = -1.0/dt * (du_dx + dv_dy);
        }
    }
    
    // Set boundary values to zero (no source at boundaries)
    for (int j = 0; j < nx; j++) {
        ptr_b[j] = 0.0;              // bottom
        ptr_b[(ny-1) * nx + j] = 0.0;  // top
    }
    
    for (int i = 0; i < ny; i++) {
        ptr_b[i * nx] = 0.0;          // left
        ptr_b[i * nx + (nx-1)] = 0.0;  // right
    }
    
    return b;
}

// Function to correct velocities based on pressure gradient
std::tuple<py::array_t<double>, py::array_t<double>> correct_velocities(
    py::array_t<double> u,       // x-velocity field
    py::array_t<double> v,       // y-velocity field
    py::array_t<double> p,       // Pressure field
    double dx,                   // Grid spacing in x-direction
    double dy,                   // Grid spacing in y-direction
    double dt                    // Time step
) {
    // Access input arrays
    py::buffer_info buf_u = u.request();
    py::buffer_info buf_v = v.request();
    py::buffer_info buf_p = p.request();
    
    if (buf_u.ndim != 2 || buf_v.ndim != 2 || buf_p.ndim != 2)
        throw std::runtime_error("Input arrays must be 2D");
    
    int ny = buf_u.shape[0];
    int nx = buf_u.shape[1];
    
    // Create output arrays
    auto u_corrected = py::array_t<double>(buf_u.shape);
    auto v_corrected = py::array_t<double>(buf_v.shape);
    
    py::buffer_info buf_u_corr = u_corrected.request();
    py::buffer_info buf_v_corr = v_corrected.request();
    
    // Get raw pointers
    double* ptr_u = static_cast<double*>(buf_u.ptr);
    double* ptr_v = static_cast<double*>(buf_v.ptr);
    double* ptr_p = static_cast<double*>(buf_p.ptr);
    double* ptr_u_corr = static_cast<double*>(buf_u_corr.ptr);
    double* ptr_v_corr = static_cast<double*>(buf_v_corr.ptr);
    
    // Copy initial values
    for (size_t i = 0; i < buf_u.size; i++) {
        ptr_u_corr[i] = ptr_u[i];
        ptr_v_corr[i] = ptr_v[i];
    }
    
    // Correct u-velocity in interior cells
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int idx = i * nx + j;
            
            // Compute pressure gradient in x-direction
            double dp_dx = (ptr_p[i * nx + (j+1)] - ptr_p[i * nx + (j-1)]) / (2.0 * dx);
            
            // Correct u velocity
            ptr_u_corr[idx] = ptr_u[idx] - dt * dp_dx;
        }
    }
    
    // Correct v-velocity in interior cells
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int idx = i * nx + j;
            
            // Compute pressure gradient in y-direction
            double dp_dy = (ptr_p[(i+1) * nx + j] - ptr_p[(i-1) * nx + j]) / (2.0 * dy);
            
            // Correct v velocity
            ptr_v_corr[idx] = ptr_v[idx] - dt * dp_dy;
        }
    }
    
    // Enforce boundary conditions for u-velocity
    // Bottom and top boundaries (no-slip)
    for (int j = 0; j < nx; j++) {
        ptr_u_corr[j] = 0.0;              // bottom
        ptr_u_corr[(ny-1) * nx + j] = 1.0;  // top (lid velocity = 1.0)
    }
    
    // Left and right boundaries (no-slip)
    for (int i = 0; i < ny; i++) {
        ptr_u_corr[i * nx] = 0.0;          // left
        ptr_u_corr[i * nx + (nx-1)] = 0.0;  // right
    }
    
    // Enforce boundary conditions for v-velocity (no-slip everywhere)
    for (int j = 0; j < nx; j++) {
        ptr_v_corr[j] = 0.0;              // bottom
        ptr_v_corr[(ny-1) * nx + j] = 0.0;  // top
    }
    
    for (int i = 0; i < ny; i++) {
        ptr_v_corr[i * nx] = 0.0;          // left
        ptr_v_corr[i * nx + (nx-1)] = 0.0;  // right
    }
    
    return std::make_tuple(u_corrected, v_corrected);
}

// Calculate the residuals
py::dict calculate_residuals(
    py::array_t<double> u,       // Current x-velocity
    py::array_t<double> v,       // Current y-velocity
    py::array_t<double> u_prev,  // Previous x-velocity
    py::array_t<double> v_prev,  // Previous y-velocity
    py::array_t<double> p        // Pressure field
) {
    py::buffer_info buf_u = u.request();
    py::buffer_info buf_v = v.request();
    py::buffer_info buf_u_prev = u_prev.request();
    py::buffer_info buf_v_prev = v_prev.request();
    
    int ny = buf_u.shape[0];
    int nx = buf_u.shape[1];
    
    double* ptr_u = static_cast<double*>(buf_u.ptr);
    double* ptr_v = static_cast<double*>(buf_v.ptr);
    double* ptr_u_prev = static_cast<double*>(buf_u_prev.ptr);
    double* ptr_v_prev = static_cast<double*>(buf_v_prev.ptr);
    double* ptr_p = static_cast<double*>(p.request().ptr);
    
    // Calculate momentum residuals
    double u_res = 0.0;
    double v_res = 0.0;
    
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int idx = i * nx + j;
            u_res += std::abs(ptr_u[idx] - ptr_u_prev[idx]);
            v_res += std::abs(ptr_v[idx] - ptr_v_prev[idx]);
        }
    }
    
    // Normalize residuals
    u_res /= ((ny-2) * (nx-2));
    v_res /= ((ny-2) * (nx-2));
    
    // Calculate continuity residual
    double cont_res = 0.0;
    for (int i = 1; i < ny-1; i++) {
        for (int j = 1; j < nx-1; j++) {
            int idx = i * nx + j;
            int idx_e = i * nx + (j+1);
            int idx_w = i * nx + (j-1);
            int idx_n = (i+1) * nx + j;
            int idx_s = (i-1) * nx + j;
            
            // Simple approximation of divergence
            double div = (ptr_u[idx_e] - ptr_u[idx_w]) + (ptr_v[idx_n] - ptr_v[idx_s]);
            cont_res += std::abs(div);
        }
    }
    
    cont_res /= ((ny-2) * (nx-2) * 2.0);  // Normalize by number of cells and 2.0 for the central difference
    
    // Return a dictionary with the residuals
    py::dict residuals;
    residuals["u_res"] = u_res;
    residuals["v_res"] = v_res;
    residuals["cont_res"] = cont_res;
    
    return residuals;
}

PYBIND11_MODULE(pyflow_core_cfd, m) {
    m.doc() = "High-performance core functions for PyFlow CFD solver (CFD module)";
    m.def("solve_pressure_poisson", &solve_pressure_poisson,
        "Solve the pressure Poisson equation using Jacobi iteration",
        py::arg("b"), py::arg("p_init"), py::arg("dx"), py::arg("dy"),
        py::arg("max_iter") = 1000, py::arg("tolerance") = 1e-4, py::arg("alpha_p") = 0.8);
    m.def("calculate_pressure_source", &calculate_pressure_source,
        "Calculate the source term for the pressure equation",
        py::arg("u"), py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("dt"));
    m.def("correct_velocities", &correct_velocities,
        "Correct velocities based on pressure gradient",
        py::arg("u"), py::arg("v"), py::arg("p"),
        py::arg("dx"), py::arg("dy"), py::arg("dt"));
    m.def("calculate_residuals", &calculate_residuals,
        "Calculate the residuals of the solution",
        py::arg("u"), py::arg("v"), py::arg("u_prev"),
        py::arg("v_prev"), py::arg("p"));
}
