import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

# Add the current directory to path to find the compiled module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyflow_core_cfd
    print("Successfully imported the C++ CFD extension!")
except ImportError as e:
    print(f"Error importing the C++ CFD extension: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    sys.exit(1)

def test_pressure_poisson_solver():
    """Test the C++ pressure Poisson solver"""
    print("\n--- Testing C++ Pressure Poisson Solver ---")
    
    # Create a grid
    n = 65  # Grid size
    L = 1.0  # Domain size
    dx = dy = L / (n - 1)  # Grid spacing
    
    # Create a simple test problem
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    
    # Create a source term (RHS of the Poisson equation)
    b = np.zeros((n, n))
    for i in range(1, n-1):
        for j in range(1, n-1):
            # Some arbitrary source pattern
            b[i, j] = np.sin(np.pi * X[i, j]) * np.sin(np.pi * Y[i, j])
    
    # Initial guess for pressure
    p_init = np.zeros((n, n))
    
    # Solve with the C++ function
    print("Solving pressure Poisson equation with C++ function...")
    start_time = time.time()
    p_cpp = pyflow_core_cfd.solve_pressure_poisson(
        b, p_init, dx, dy, 
        max_iter=1000, tolerance=1e-6, alpha_p=0.8
    )
    cpp_time = time.time() - start_time
    print(f"C++ solution time: {cpp_time:.3f} seconds")
    
    # Visualize the solution
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, p_cpp, cmap='viridis', levels=50)
    plt.colorbar(label='Pressure')
    plt.title('Pressure Field (C++ Solution)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("pressure_cpp_solution.png", dpi=300)
    plt.close()
    
    return p_cpp, cpp_time

def test_velocity_correction():
    """Test the C++ velocity correction function"""
    print("\n--- Testing C++ Velocity Correction ---")
    
    # Create a grid
    n = 65  # Grid size
    L = 1.0  # Domain size
    dx = dy = L / (n - 1)  # Grid spacing
    dt = 0.001  # Time step
    
    # Create initial velocity fields for a lid-driven cavity
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    
    # Set lid velocity (top boundary)
    u[-1, 1:-1] = 1.0
    
    # Create a pressure field (some arbitrary pattern)
    p = np.zeros((n, n))
    for i in range(1, n-1):
        for j in range(1, n-1):
            # Some pattern that will create pressure gradients
            p[i, j] = 0.1 * np.sin(2 * np.pi * i / n) * np.cos(2 * np.pi * j / n)
    
    # Correct velocities using the C++ function
    print("Correcting velocities with C++ function...")
    start_time = time.time()
    u_corr, v_corr = pyflow_core_cfd.correct_velocities(u, v, p, dx, dy, dt)
    cpp_time = time.time() - start_time
    print(f"C++ velocity correction time: {cpp_time:.3f} seconds")
    
    # Calculate the change in velocities
    u_diff = u_corr - u
    v_diff = v_corr - v
    
    # Plot the velocity corrections
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, p, cmap='viridis', levels=20)
    plt.colorbar(label='Pressure')
    plt.title('Pressure Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, u_diff, cmap='RdBu_r', levels=20)
    plt.colorbar(label='u correction')
    plt.title('U-Velocity Correction')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, v_diff, cmap='RdBu_r', levels=20)
    plt.colorbar(label='v correction')
    plt.title('V-Velocity Correction')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig("velocity_correction.png", dpi=300)
    plt.close()
    
    return u_corr, v_corr, cpp_time

def test_residuals_calculation():
    """Test the C++ residuals calculation function"""
    print("\n--- Testing C++ Residuals Calculation ---")
    
    # Create a grid
    n = 65  # Grid size
    
    # Create velocity fields
    u_prev = np.random.rand(n, n) * 0.1  # Previous velocity
    v_prev = np.random.rand(n, n) * 0.1
    
    # Create slightly different current velocity fields (simulating a timestep)
    u = u_prev + np.random.rand(n, n) * 0.01
    v = v_prev + np.random.rand(n, n) * 0.01
    
    # Set boundary conditions
    u[0, :] = 0.0  # Bottom
    u[-1, :] = 1.0  # Top (lid)
    u[:, 0] = u[:, -1] = 0.0  # Left and right
    
    v[0, :] = v[-1, :] = 0.0  # Bottom and top
    v[:, 0] = v[:, -1] = 0.0  # Left and right
    
    # Create a pressure field
    p = np.zeros((n, n))
    
    # Calculate residuals using the C++ function
    print("Calculating residuals with C++ function...")
    start_time = time.time()
    residuals = pyflow_core_cfd.calculate_residuals(u, v, u_prev, v_prev, p)
    cpp_time = time.time() - start_time
    print(f"C++ residuals calculation time: {cpp_time:.3f} seconds")
    
    # Print the residuals
    print(f"u-momentum residual: {residuals['u_res']:.8f}")
    print(f"v-momentum residual: {residuals['v_res']:.8f}")
    print(f"continuity residual: {residuals['cont_res']:.8f}")
    
    return residuals, cpp_time

if __name__ == "__main__":
    # Test the C++ pressure Poisson solver
    p_solution, pressure_time = test_pressure_poisson_solver()
    
    # Test the velocity correction function
    u_corr, v_corr, correction_time = test_velocity_correction()
    
    # Test the residuals calculation
    residuals, residuals_time = test_residuals_calculation()
    
    print("\n--- Summary of C++ Function Performance ---")
    print(f"Pressure Poisson solver: {pressure_time:.3f} seconds")
    print(f"Velocity correction:     {correction_time:.3f} seconds")
    print(f"Residuals calculation:   {residuals_time:.3f} seconds")
