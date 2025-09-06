"""
Example script demonstrating how to use the PyFlow hybrid C++/Python solver
for the lid-driven cavity problem.

This example shows how to:
1. Set up and run a simulation
2. Monitor convergence through residuals
3. Visualize the results
4. Compare with pure Python implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import PyFlow modules
from pyflow.solver import solve_lid_driven_cavity as solve_python
from pyflow.hybrid_solver import solve_lid_driven_cavity as solve_hybrid
from pyflow.grid import Grid
from pyflow.logging import LiveLogger

def run_example():
    """Run and visualize lid-driven cavity flow using hybrid solver"""
    
    # Set problem parameters
    N = 65          # Grid size (N×N)
    L = 1.0         # Domain size
    Re = 400        # Reynolds number (try 100, 400, or 1000)
    dt = 0.001      # Time step
    T = 5.0         # Total simulation time
    
    print(f"\n{'='*60}")
    print(f"PyFlow Hybrid Solver Example - Lid-Driven Cavity")
    print(f"{'='*60}")
    print(f"Grid: {N}×{N}")
    print(f"Reynolds number: {Re}")
    print(f"Time step: {dt}")
    print(f"Total simulation time: {T}")
    print(f"{'='*60}\n")
    
    # Create grid
    grid = Grid(N, L)
    dx = dy = grid.dx
    
    # Create loggers
    log_interval = 100
    logger_py = LiveLogger(N, Re, dt, T, log_interval)
    logger_hybrid = LiveLogger(N, Re, dt, T, log_interval)
    
    # Run pure Python solver
    print("\nRunning pure Python solver...")
    start_time = time.time()
    u_py, v_py, p_py, res_py = solve_python(
        N, dx, dy, Re, dt, T,
        p_iterations=500,
        logger=logger_py
    )
    py_time = time.time() - start_time
    print(f"\nPython solver completed in {py_time:.3f} seconds")
    
    # Run hybrid solver
    print("\nRunning hybrid C++/Python solver...")
    start_time = time.time()
    u_hybrid, v_hybrid, p_hybrid, res_hybrid = solve_hybrid(
        N, dx, dy, Re, dt, T,
        p_iterations=500,
        logger=logger_hybrid
    )
    hybrid_time = time.time() - start_time
    print(f"\nHybrid solver completed in {hybrid_time:.3f} seconds")
    
    # Calculate speedup
    speedup = py_time / hybrid_time
    print(f"\nHybrid solver speedup: {speedup:.2f}×")
    
    # Compare solutions
    max_diff_u = np.max(np.abs(u_py - u_hybrid))
    max_diff_v = np.max(np.abs(v_py - v_hybrid))
    max_diff_p = np.max(np.abs(p_py - p_hybrid))
    
    print(f"\nSolution comparison - Maximum absolute differences:")
    print(f"u-velocity: {max_diff_u:.8f}")
    print(f"v-velocity: {max_diff_v:.8f}")
    print(f"pressure:   {max_diff_p:.8f}")
    
    # Create plots
    print("\nCreating visualizations...")
    
    # Plot residual history
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(res_py['u_res'], 'b-', label='u-momentum')
    plt.semilogy(res_py['v_res'], 'r-', label='v-momentum')
    plt.semilogy(res_py['cont_res'], 'g-', label='continuity')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Pure Python Solver')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(res_hybrid['u_res'], 'b-', label='u-momentum')
    plt.semilogy(res_hybrid['v_res'], 'r-', label='v-momentum')
    plt.semilogy(res_hybrid['cont_res'], 'g-', label='continuity')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Hybrid Solver')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'hybrid_example_Re{Re}_residuals.png', dpi=300)
    
    # Plot solution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.contourf(grid.X, grid.Y, u_hybrid, levels=30, cmap='viridis')
    plt.colorbar(label='u-velocity')
    plt.streamplot(grid.X.T, grid.Y.T, u_hybrid.T, v_hybrid.T, color='white', linewidth=0.5, density=2)
    plt.title(f'Lid-Driven Cavity Flow (Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 3, 2)
    plt.contourf(grid.X, grid.Y, p_hybrid, levels=30, cmap='coolwarm')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Compute and plot vorticity
    vorticity = np.zeros_like(u_hybrid)
    for i in range(1, N-1):
        for j in range(1, N-1):
            vorticity[i, j] = (v_hybrid[i, j+1] - v_hybrid[i, j-1]) / (2*dx) - \
                              (u_hybrid[i+1, j] - u_hybrid[i-1, j]) / (2*dy)
    
    plt.subplot(1, 3, 3)
    plt.contourf(grid.X, grid.Y, vorticity, levels=30, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity Field')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig(f'hybrid_example_Re{Re}_solution.png', dpi=300)
    
    # Plot centerline velocities
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(u_hybrid[:, N//2], grid.y, 'b-', label='Hybrid')
    plt.plot(u_py[:, N//2], grid.y, 'r--', label='Python')
    plt.xlabel('u-velocity')
    plt.ylabel('y')
    plt.title('Vertical Centerline u-velocity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(grid.x, v_hybrid[N//2, :], 'b-', label='Hybrid')
    plt.plot(grid.x, v_py[N//2, :], 'r--', label='Python')
    plt.xlabel('x')
    plt.ylabel('v-velocity')
    plt.title('Horizontal Centerline v-velocity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'hybrid_example_Re{Re}_centerline.png', dpi=300)
    
    print("\nVisualizations saved as:")
    print(f"- hybrid_example_Re{Re}_residuals.png")
    print(f"- hybrid_example_Re{Re}_solution.png")
    print(f"- hybrid_example_Re{Re}_centerline.png")
    
    print(f"\n{'='*60}")
    print(f"Example completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_example()
