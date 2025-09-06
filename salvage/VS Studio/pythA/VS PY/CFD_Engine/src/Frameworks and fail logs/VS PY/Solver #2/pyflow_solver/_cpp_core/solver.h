#pragma once
#include <vector>
#include <tuple>

namespace pyflow {

class CFDSolver {
public:
    CFDSolver(int n_points, double dx, double dy, double reynolds);
    
    void simulation_step(double dt);
    
    void solve_lid_driven_cavity(double dt, double total_time, int p_iterations = 1000);
    
    // Getters for solution fields
    const std::vector<std::vector<double>>& get_u() const;
    const std::vector<std::vector<double>>& get_v() const;
    const std::vector<std::vector<double>>& get_p() const;
    const std::vector<double>& get_u_residuals() const;
    const std::vector<double>& get_v_residuals() const;
    const std::vector<double>& get_cont_residuals() const;
    
private:
    // Grid dimensions
    int n_points_;
    double dx_;
    double dy_;
    
    // Physical parameters
    double reynolds_;
    
    // Solution fields
    std::vector<std::vector<double>> u_;
    std::vector<std::vector<double>> v_;
    std::vector<std::vector<double>> p_;
    
    // Intermediate fields
    std::vector<std::vector<double>> u_star_;
    std::vector<std::vector<double>> v_star_;
    
    // Residuals history
    std::vector<double> u_residuals_;
    std::vector<double> v_residuals_;
    std::vector<double> cont_residuals_;
    
    // Helper methods
    void initialize_fields();
    void set_boundary_conditions();
    void calculate_intermediate_velocities(double dt);
    void solve_pressure_poisson(double dt, int max_iterations, double tolerance = 1e-5);
    void correct_velocities(double dt);
    std::tuple<double, double, double> calculate_residuals();
    
    // Under-relaxation factors
    double alpha_u_ = 0.8;
    double alpha_p_ = 0.5;
};

} // namespace pyflow
