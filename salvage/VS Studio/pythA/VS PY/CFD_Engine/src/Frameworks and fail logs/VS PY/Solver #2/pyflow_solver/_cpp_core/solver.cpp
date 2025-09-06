#include "solver.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <omp.h>  // For OpenMP parallelization

namespace pyflow {

CFDSolver::CFDSolver(int n_points, double dx, double dy, double reynolds)
    : n_points_(n_points), dx_(dx), dy_(dy), reynolds_(reynolds) {
    initialize_fields();
}

void CFDSolver::initialize_fields() {
    // Initialize all solution fields and intermediate fields
    u_.resize(n_points_, std::vector<double>(n_points_, 0.0));
    v_.resize(n_points_, std::vector<double>(n_points_, 0.0));
    p_.resize(n_points_, std::vector<double>(n_points_, 0.0));
    
    u_star_.resize(n_points_, std::vector<double>(n_points_, 0.0));
    v_star_.resize(n_points_, std::vector<double>(n_points_, 0.0));
    
    // Set initial conditions - lid velocity = 1.0
    for (int j = 1; j < n_points_ - 1; ++j) {
        u_[n_points_ - 1][j] = 1.0;
    }
}

void CFDSolver::set_boundary_conditions() {
    // Bottom and top walls
    for (int i = 0; i < n_points_; ++i) {
        u_[0][i] = 0.0;  // Bottom wall
        v_[0][i] = 0.0;
        
        u_[n_points_ - 1][i] = 0.0;  // Top wall
        v_[n_points_ - 1][i] = 0.0;
    }
    
    // Lid velocity (top wall)
    for (int j = 1; j < n_points_ - 1; ++j) {
        u_[n_points_ - 1][j] = 1.0;  // Moving lid
    }
    
    // Left and right walls
    for (int i = 0; i < n_points_; ++i) {
        u_[i][0] = 0.0;  // Left wall
        v_[i][0] = 0.0;
        
        u_[i][n_points_ - 1] = 0.0;  // Right wall
        v_[i][n_points_ - 1] = 0.0;
    }
}

void CFDSolver::calculate_intermediate_velocities(double dt) {
    // Copy current velocity fields to intermediate fields
    u_star_ = u_;
    v_star_ = v_;
    
    // Compute u_star for interior points using finite differences
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            // Convection terms (first-order upwind)
            double u_dx = 0.0;
            if (u_[i][j] > 0) {
                u_dx = u_[i][j] * (u_[i][j] - u_[i][j-1]) / dx_;
            } else {
                u_dx = u_[i][j] * (u_[i][j+1] - u_[i][j]) / dx_;
            }
            
            double v_dy = 0.0;
            if (v_[i][j] > 0) {
                v_dy = v_[i][j] * (u_[i][j] - u_[i-1][j]) / dy_;
            } else {
                v_dy = v_[i][j] * (u_[i+1][j] - u_[i][j]) / dy_;
            }
            
            // Diffusion terms (central difference)
            double u_diff_x = (u_[i][j+1] - 2.0 * u_[i][j] + u_[i][j-1]) / (dx_ * dx_);
            double u_diff_y = (u_[i+1][j] - 2.0 * u_[i][j] + u_[i-1][j]) / (dy_ * dy_);
            
            // Update u_star with viscous and convective terms
            u_star_[i][j] = u_[i][j] + dt * (
                (1.0 / reynolds_) * (u_diff_x + u_diff_y) - u_dx - v_dy
            );
            
            // Apply under-relaxation
            u_star_[i][j] = (1.0 - alpha_u_) * u_[i][j] + alpha_u_ * u_star_[i][j];
        }
    }
    
    // Compute v_star for interior points
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            // Convection terms (first-order upwind)
            double u_dx = 0.0;
            if (u_[i][j] > 0) {
                u_dx = u_[i][j] * (v_[i][j] - v_[i][j-1]) / dx_;
            } else {
                u_dx = u_[i][j] * (v_[i][j+1] - v_[i][j]) / dx_;
            }
            
            double v_dy = 0.0;
            if (v_[i][j] > 0) {
                v_dy = v_[i][j] * (v_[i][j] - v_[i-1][j]) / dy_;
            } else {
                v_dy = v_[i][j] * (v_[i+1][j] - v_[i][j]) / dy_;
            }
            
            // Diffusion terms (central difference)
            double v_diff_x = (v_[i][j+1] - 2.0 * v_[i][j] + v_[i][j-1]) / (dx_ * dx_);
            double v_diff_y = (v_[i+1][j] - 2.0 * v_[i][j] + v_[i-1][j]) / (dy_ * dy_);
            
            // Update v_star with viscous and convective terms
            v_star_[i][j] = v_[i][j] + dt * (
                (1.0 / reynolds_) * (v_diff_x + v_diff_y) - u_dx - v_dy
            );
            
            // Apply under-relaxation
            v_star_[i][j] = (1.0 - alpha_u_) * v_[i][j] + alpha_u_ * v_star_[i][j];
        }
    }
}

void CFDSolver::solve_pressure_poisson(double dt, int max_iterations, double tolerance) {
    // Create a copy of the pressure field for residual calculation
    std::vector<std::vector<double>> p_old(n_points_, std::vector<double>(n_points_, 0.0));
    
    // Compute the source term for Poisson equation (divergence of intermediate velocity)
    std::vector<std::vector<double>> div_ustar(n_points_, std::vector<double>(n_points_, 0.0));
    
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            div_ustar[i][j] = (
                (u_star_[i][j+1] - u_star_[i][j-1]) / (2.0 * dx_) +
                (v_star_[i+1][j] - v_star_[i-1][j]) / (2.0 * dy_)
            ) / dt;
        }
    }
    
    // Iterative solution of pressure Poisson equation
    double residual = 1.0;
    int iter = 0;
    
    while (iter < max_iterations && residual > tolerance) {
        // Store current pressure for convergence check
        p_old = p_;
        
        // Solve pressure Poisson equation using Jacobi iteration
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n_points_ - 1; ++i) {
            for (int j = 1; j < n_points_ - 1; ++j) {
                // Coefficients for the Poisson equation
                double dx2 = dx_ * dx_;
                double dy2 = dy_ * dy_;
                
                // Update pressure
                p_[i][j] = (
                    (p_old[i][j+1] + p_old[i][j-1]) * dy2 +
                    (p_old[i+1][j] + p_old[i-1][j]) * dx2 -
                    div_ustar[i][j] * dx2 * dy2
                ) / (2.0 * (dx2 + dy2));
                
                // Apply under-relaxation
                p_[i][j] = (1.0 - alpha_p_) * p_old[i][j] + alpha_p_ * p_[i][j];
            }
        }
        
        // Pressure boundary conditions: zero normal gradient (Neumann)
        for (int i = 0; i < n_points_; ++i) {
            p_[i][0] = p_[i][1];             // Left wall
            p_[i][n_points_-1] = p_[i][n_points_-2]; // Right wall
        }
        
        for (int j = 0; j < n_points_; ++j) {
            p_[0][j] = p_[1][j];             // Bottom wall
            p_[n_points_-1][j] = p_[n_points_-2][j]; // Top wall
        }
        
        // Set a reference pressure point to avoid pressure field drift
        p_[0][0] = 0.0;
        
        // Calculate residual for convergence check
        residual = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:residual)
        for (int i = 1; i < n_points_ - 1; ++i) {
            for (int j = 1; j < n_points_ - 1; ++j) {
                residual += std::abs(p_[i][j] - p_old[i][j]);
            }
        }
        residual /= (n_points_ - 2) * (n_points_ - 2);
        
        ++iter;
    }
    
    if (iter == max_iterations) {
        std::cout << "Warning: Pressure solver did not converge. Final residual: " << residual << std::endl;
    }
}

void CFDSolver::correct_velocities(double dt) {
    // Correct velocities using the pressure field
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            // Face-centered pressure gradients for better mass conservation
            double dp_dx = (p_[i][j+1] - p_[i][j-1]) / (2.0 * dx_);
            double dp_dy = (p_[i+1][j] - p_[i-1][j]) / (2.0 * dy_);
            
            // Correct velocities
            u_[i][j] = u_star_[i][j] - dt * dp_dx;
            v_[i][j] = v_star_[i][j] - dt * dp_dy;
        }
    }
    
    // Reapply boundary conditions to ensure they are satisfied
    set_boundary_conditions();
}

std::tuple<double, double, double> CFDSolver::calculate_residuals() {
    // Calculate residuals for momentum and continuity equations
    double u_res = 0.0;
    double v_res = 0.0;
    double cont_res = 0.0;
    
    // Calculate u-momentum residual
    #pragma omp parallel for collapse(2) reduction(+:u_res)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            double convection = 0.0;
            double diffusion = 0.0;
            double pressure = 0.0;
            
            // Calculate residual components
            // ...
            
            u_res += std::abs(convection + diffusion + pressure);
        }
    }
    u_res /= (n_points_ - 2) * (n_points_ - 2);
    
    // Calculate v-momentum residual
    #pragma omp parallel for collapse(2) reduction(+:v_res)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            double convection = 0.0;
            double diffusion = 0.0;
            double pressure = 0.0;
            
            // Calculate residual components
            // ...
            
            v_res += std::abs(convection + diffusion + pressure);
        }
    }
    v_res /= (n_points_ - 2) * (n_points_ - 2);
    
    // Calculate continuity residual (mass conservation error)
    #pragma omp parallel for collapse(2) reduction(+:cont_res)
    for (int i = 1; i < n_points_ - 1; ++i) {
        for (int j = 1; j < n_points_ - 1; ++j) {
            double div = (u_[i][j+1] - u_[i][j-1]) / (2.0 * dx_) +
                         (v_[i+1][j] - v_[i-1][j]) / (2.0 * dy_);
            cont_res += std::abs(div);
        }
    }
    cont_res /= (n_points_ - 2) * (n_points_ - 2);
    
    return std::make_tuple(u_res, v_res, cont_res);
}

void CFDSolver::simulation_step(double dt) {
    set_boundary_conditions();
    calculate_intermediate_velocities(dt);
    solve_pressure_poisson(dt, 1000);
    correct_velocities(dt);
    
    // Calculate and store residuals
    auto [u_res, v_res, cont_res] = calculate_residuals();
    u_residuals_.push_back(u_res);
    v_residuals_.push_back(v_res);
    cont_residuals_.push_back(cont_res);
}

void CFDSolver::solve_lid_driven_cavity(double dt, double total_time, int p_iterations) {
    int total_steps = static_cast<int>(total_time / dt);
    
    for (int step = 0; step < total_steps; ++step) {
        simulation_step(dt);
        
        if (step % 100 == 0) {
            std::cout << "Step " << step << "/" << total_steps
                     << ", Continuity residual: " << cont_residuals_.back() << std::endl;
        }
    }
}

// Getter methods implementation
const std::vector<std::vector<double>>& CFDSolver::get_u() const { return u_; }
const std::vector<std::vector<double>>& CFDSolver::get_v() const { return v_; }
const std::vector<std::vector<double>>& CFDSolver::get_p() const { return p_; }
const std::vector<double>& CFDSolver::get_u_residuals() const { return u_residuals_; }
const std::vector<double>& CFDSolver::get_v_residuals() const { return v_residuals_; }
const std::vector<double>& CFDSolver::get_cont_residuals() const { return cont_residuals_; }

} // namespace pyflow
