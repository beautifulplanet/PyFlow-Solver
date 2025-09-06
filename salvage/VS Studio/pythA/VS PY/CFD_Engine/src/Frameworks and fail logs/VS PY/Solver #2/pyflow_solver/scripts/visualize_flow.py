import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pyflow.solver import solve_lid_driven_cavity
from pyflow.logging import LiveLogger
from pyflow.grid import Grid

def find_stream_function(u, v, dx, dy):
    """
    Calculate the stream function from velocity components.
    Uses the fact that u = dψ/dy and v = -dψ/dx
    Returns the stream function array.
    """
    ny, nx = u.shape
    psi = np.zeros((ny, nx))
    
    # Integrate from bottom to top
    for j in range(1, ny):
        psi[j, :] = psi[j-1, :] + u[j-1, :] * dy
    
    return psi

def find_vorticity(u, v, dx, dy):
    """
    Calculate vorticity (curl of velocity)
    ω = ∂v/∂x - ∂u/∂y
    """
    ny, nx = u.shape
    omega = np.zeros((ny, nx))
    
    # Central differencing for interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            dudv = (v[i, j+1] - v[i, j-1]) / (2*dx)
            dvdu = (u[i+1, j] - u[i-1, j]) / (2*dy)
            omega[i, j] = dudv - dvdu
    
    return omega

def run_simulation(Re, NPOINTS, T, output_dir):
    """Run simulation and generate visualization"""
    L = 1.0
    grid = Grid(NPOINTS, L)
    
    # Calculate time step based on stability requirements
    h = L / (NPOINTS - 1)
    dt = min(0.001, 0.25 * h * h)  # Adaptive time step
    
    print(f"Running simulation with Re={Re}, grid={NPOINTS}x{NPOINTS}, dt={dt:.6f}")
    logger = LiveLogger(NPOINTS, Re, dt, T, log_interval=100)
    
    u, v, p, residuals = solve_lid_driven_cavity(
        grid.NPOINTS, grid.dx, grid.dy, Re, dt, T,
        logger=logger
    )
    
    # Calculate derived quantities
    psi = find_stream_function(u, v, grid.dx, grid.dy)
    omega = find_vorticity(u, v, grid.dx, grid.dy)
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create output directory
    output_path = Path(output_dir) / f"Re{Re}_N{NPOINTS}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot velocity vectors
    plt.figure(figsize=(10, 10))
    # Skip some points for clarity
    skip = max(1, NPOINTS // 25)
    plt.quiver(grid.X[::skip, ::skip], grid.Y[::skip, ::skip], 
               u[::skip, ::skip], v[::skip, ::skip], 
               vel_mag[::skip, ::skip], cmap=cm.viridis)
    plt.colorbar(label='Velocity Magnitude')
    plt.title(f'Velocity Field for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(output_path / "velocity_vectors.png", dpi=300)
    plt.close()
    
    # Plot streamlines
    plt.figure(figsize=(10, 10))
    levels = np.linspace(np.min(psi), np.max(psi), 30)
    contour = plt.contour(grid.X, grid.Y, psi, levels=levels, cmap=cm.RdBu_r)
    plt.colorbar(contour, label='Stream Function')
    plt.title(f'Streamlines for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(output_path / "streamlines.png", dpi=300)
    plt.close()
    
    # Plot vorticity contours
    plt.figure(figsize=(10, 10))
    levels = np.linspace(np.min(omega), np.max(omega), 30)
    contour = plt.contourf(grid.X, grid.Y, omega, levels=levels, cmap=cm.RdBu_r)
    plt.colorbar(contour, label='Vorticity')
    plt.title(f'Vorticity for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(output_path / "vorticity.png", dpi=300)
    plt.close()
    
    # Plot pressure contours
    plt.figure(figsize=(10, 10))
    levels = np.linspace(np.min(p), np.max(p), 30)
    contour = plt.contourf(grid.X, grid.Y, p, levels=levels, cmap=cm.viridis)
    plt.colorbar(contour, label='Pressure')
    plt.title(f'Pressure Field for Re = {Re}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(output_path / "pressure.png", dpi=300)
    plt.close()
    
    # Plot centerline velocity profiles
    center_idx = NPOINTS // 2
    
    # U-velocity along vertical centerline
    plt.figure(figsize=(8, 10))
    plt.plot(u[:, center_idx], grid.Y[:, center_idx], 'b-', linewidth=2)
    plt.xlabel('u-velocity')
    plt.ylabel('y-coordinate')
    plt.title(f'U-Velocity along Vertical Centerline (x=0.5), Re={Re}')
    plt.grid(True)
    plt.savefig(output_path / "u_centerline.png", dpi=300)
    plt.close()
    
    # V-velocity along horizontal centerline
    plt.figure(figsize=(10, 8))
    plt.plot(grid.X[center_idx, :], v[center_idx, :], 'r-', linewidth=2)
    plt.xlabel('x-coordinate')
    plt.ylabel('v-velocity')
    plt.title(f'V-Velocity along Horizontal Centerline (y=0.5), Re={Re}')
    plt.grid(True)
    plt.savefig(output_path / "v_centerline.png", dpi=300)
    plt.close()
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals['u_res'], 'b-', label='u-momentum')
    plt.semilogy(residuals['v_res'], 'r-', label='v-momentum')
    plt.semilogy(residuals['cont_res'], 'g-', label='continuity')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title(f'Residual History for Re={Re}')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path / "residuals.png", dpi=300)
    plt.close()
    
    # Save the final data as numpy arrays
    np.save(output_path / "u_velocity.npy", u)
    np.save(output_path / "v_velocity.npy", v)
    np.save(output_path / "pressure.npy", p)
    np.save(output_path / "stream_function.npy", psi)
    np.save(output_path / "vorticity.npy", omega)
    
    print(f"Visualization outputs saved to {output_path}")
    
    return u, v, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFD simulation and generate visualizations.")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--grid", type=int, default=65, help="Grid size (N x N)")
    parser.add_argument("--time", type=float, default=20.0, help="Total simulation time")
    parser.add_argument("--output", type=str, default="visualization_results", help="Output directory")
    args = parser.parse_args()
    
    run_simulation(args.re, args.grid, args.time, args.output)
