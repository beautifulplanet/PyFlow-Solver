import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pyflow.solver import solve_lid_driven_cavity
from pyflow.grid import Grid

def analyze_cfl_condition(u, v, dx, dy, dt):
    """
    Calculate CFL number for given velocity fields and grid spacing
    
    Parameters:
    -----------
    u, v : numpy arrays
        Velocity field components
    dx, dy : float
        Grid spacing
    dt : float
        Time step
        
    Returns:
    --------
    max_cfl : float
        Maximum CFL number
    avg_cfl : float
        Average CFL number
    """
    # Calculate CFL number in x and y directions
    cfl_x = np.abs(u) * dt / dx
    cfl_y = np.abs(v) * dt / dy
    
    # Total CFL is sum of components
    cfl_total = cfl_x + cfl_y
    
    return np.max(cfl_total), np.mean(cfl_total)

def run_simulation_with_cfl(Re=100, N=41, T=0.1):
    """
    Run a single step of simulation and analyze CFL condition
    """
    # Set up grid
    L = 1.0
    grid = Grid(N, L)
    dx, dy = grid.dx, grid.dy
    
    # Initial conditions
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    
    # Set lid velocity
    u[-1, 1:-1] = 1.0
    
    # Initial time step - will be adjusted by CFL
    dt_init = 0.001
    
    # Under-relaxation factors based on Re
    if Re <= 100:
        alpha_u, alpha_p = 0.8, 0.5
    elif Re <= 400:
        alpha_u, alpha_p = 0.7, 0.3
    else:
        alpha_u, alpha_p = 0.5, 0.2
    
    # Pressure iterations based on Re
    if Re <= 100:
        p_iterations = 50
    elif Re <= 400:
        p_iterations = 100
    else:
        p_iterations = 200
    
    # Calculate viscosity
    nu = 1.0 / Re
    
    print(f"Running CFL analysis for Re={Re}")
    print(f"Grid size: {N}x{N}, dx={dx:.5f}, dy={dy:.5f}")
    print(f"Under-relaxation: alpha_u={alpha_u}, alpha_p={alpha_p}")
    
    # Run simulation for a few steps to develop flow
    for step in range(5):
        # Analyze CFL before step
        if step > 0:  # Skip first step when velocities are 0
            max_cfl, avg_cfl = analyze_cfl_condition(u, v, dx, dy, dt_init)
            print(f"Step {step}: Max CFL={max_cfl:.4f}, Avg CFL={avg_cfl:.4f}")
            
            # Calculate adaptive time step for next iteration
            if max_cfl > 0:
                cfl_target = 0.5  # Conservative target
                dt_cfl = cfl_target * dt_init / max_cfl
                print(f"  Recommended dt: {dt_cfl:.6f}")
            
        # Update simulation for one step
        from pyflow.solver import _simulation_step
        u, v, p = _simulation_step(N, dx, dy, dt_init, nu, u, v, p, p_iterations, alpha_u, alpha_p)
    
    # Create velocity magnitude plot
    plt.figure(figsize=(7, 6))
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create mesh grid for plotting
    x = np.linspace(0, grid.Lx, grid.Nx)
    y = np.linspace(0, grid.Ly, grid.Ny)
    X, Y = np.meshgrid(x, y)
    
    # Create a custom colormap
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1), 
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('velocity', colors, N=100)
    
    # Plot velocity magnitude
    plt.contourf(X, Y, vel_mag, 20, cmap=cmap)
    plt.colorbar(label='Velocity Magnitude')
    
    # Add streamlines
    plt.streamplot(X, Y, u, v, color='white', density=1.0, 
                  linewidth=0.8, arrowsize=1)
    
    plt.title(f'Velocity Field and Streamlines (Re={Re})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(f'cfl_analysis_re{Re}.png', dpi=300)
    
    return u, v, p

if __name__ == "__main__":
    # Run CFL analysis for different Reynolds numbers
    for Re in [100, 400, 1000]:
        run_simulation_with_cfl(Re=Re)
        print("-" * 50)
