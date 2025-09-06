import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import argparse
from tabulate import tabulate  # You may need to install this: pip install tabulate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pyflow.solver import solve_lid_driven_cavity
from pyflow.logging import LiveLogger
from pyflow.grid import Grid

# Ghia et al. (1982) benchmark data
GHIA_DATA = {
    100: {
        'y': [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0],
        'u': [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.2109, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0.0]
    }
}

def calculate_vortex_center(u, v, grid):
    """
    Calculate the location of the primary vortex center by finding the point
    where both velocity components are closest to zero in the interior of the domain.
    Returns (x, y) coordinates and velocity magnitude at the vortex center.
    """
    # Create velocity magnitude field
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Only consider interior points (avoid boundaries)
    interior_slice = slice(1, -1), slice(1, -1)
    interior_vel_mag = vel_mag[interior_slice]
    
    # Find the index of minimum velocity magnitude
    min_idx = np.unravel_index(np.argmin(interior_vel_mag), interior_vel_mag.shape)
    
    # Convert to grid coordinates
    x_idx = min_idx[1] + 1  # Add 1 because we sliced from 1:-1
    y_idx = min_idx[0] + 1
    
    x = grid.X[y_idx, x_idx]
    y = grid.Y[y_idx, x_idx]
    min_vel = vel_mag[y_idx, x_idx]
    
    return x, y, min_vel

def calculate_rmse(predicted, actual):
    """Calculate Root Mean Square Error between two arrays"""
    return np.sqrt(np.mean((predicted - actual)**2))

def calculate_order_of_convergence(grids, errors):
    """
    Calculate observed order of convergence using Richardson extrapolation.
    For a method of order p, error ≈ C * h^p, where h is grid spacing.
    """
    if len(grids) < 3 or len(errors) < 3:
        return None, None
    
    results = []
    
    for i in range(len(grids)-2):
        h1 = 1 / (grids[i] - 1)
        h2 = 1 / (grids[i+1] - 1)
        h3 = 1 / (grids[i+2] - 1)
        
        e1 = errors[i]
        e2 = errors[i+1]
        e3 = errors[i+2]
        
        # Skip if errors aren't decreasing monotonically
        if not (e1 > e2 > e3):
            continue
            
        # Grid refinement ratios
        r1 = h1 / h2
        r2 = h2 / h3
        
        # Calculate observed order of convergence
        if abs(e1 - e2) > 1e-10 and abs(e2 - e3) > 1e-10:
            p = np.log((e1 - e2) / (e2 - e3)) / np.log(r1)
            
            # Extrapolated value at zero grid spacing
            phi_ext = e3 + (e3 - e2) / (r2**p - 1)
            
            # Grid Convergence Index (GCI)
            safety_factor = 1.25
            gci_fine = safety_factor * abs(e3 - e2) / (r2**p - 1)
            
            results.append({
                'grids': (grids[i], grids[i+1], grids[i+2]),
                'order': p,
                'extrapolated': phi_ext,
                'gci': gci_fine
            })
    
    if not results:
        return None, None
    
    # Return the result with the most reasonable order of convergence
    results.sort(key=lambda x: abs(x['order'] - 2))  # Sort by closeness to 2 (theoretical order)
    return results[0]['order'], results[0]['extrapolated']

def run_grid_study(Re, grid_sizes, sim_time, output_dir):
    """Run simulations at different grid resolutions and analyze convergence"""
    L = 1.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    u_rmse_values = []
    v_center_values = []
    vortex_x_values = []
    vortex_y_values = []
    vortex_strength_values = []
    grid_spacings = []
    
    for NPOINTS in grid_sizes:
        grid = Grid(NPOINTS, L)
        h = L / (NPOINTS - 1)  # Grid spacing
        grid_spacings.append(h)
        
        # Calculate time step based on grid size (for numerical stability)
        dt = min(0.001, 0.25 * h**2)
        
        print(f"\nRunning simulation with {NPOINTS}×{NPOINTS} grid (h={h:.6f}), Re={Re}, dt={dt:.6f}")
        logger = LiveLogger(NPOINTS, Re, dt, sim_time, log_interval=100)
        
        u, v, p, residuals = solve_lid_driven_cavity(
            grid.NPOINTS, grid.dx, grid.dy, Re, dt, sim_time,
            logger=logger
        )
        
        # Calculate vortex center
        vortex_x, vortex_y, vortex_strength = calculate_vortex_center(u, v, grid)
        vortex_x_values.append(vortex_x)
        vortex_y_values.append(vortex_y)
        vortex_strength_values.append(vortex_strength)
        
        # Calculate centerline v-velocity at x=0.5, y=0.5
        center_idx = NPOINTS // 2
        v_center = v[center_idx, center_idx]
        v_center_values.append(v_center)
        
        # Compare with Ghia benchmark data if Re=100
        if Re == 100:
            # Extract centerline velocities
            center_idx = NPOINTS // 2
            u_centerline = u[:, center_idx]
            y_coords = grid.Y[:, center_idx]
            
            # Interpolate solution to Ghia data points
            ghia_y = GHIA_DATA[Re]['y']
            ghia_u = GHIA_DATA[Re]['u']
            u_interp = np.interp(ghia_y, y_coords, u_centerline)
            
            # Calculate error
            u_rmse = calculate_rmse(u_interp, ghia_u)
            u_rmse_values.append(u_rmse)
            
            print(f"  RMSE against Ghia et al.: {u_rmse:.6f}")
        else:
            u_rmse_values.append(None)
        
        print(f"  Vortex center: ({vortex_x:.6f}, {vortex_y:.6f})")
        print(f"  V-velocity at center: {v_center:.6f}")
        
        results.append({
            'Grid': f"{NPOINTS}×{NPOINTS}",
            'h': h,
            'Vortex X': vortex_x,
            'Vortex Y': vortex_y,
            'Center V': v_center,
            'RMSE': u_rmse_values[-1]
        })
    
    # Calculate observed order of convergence
    p_vortex_x, ext_vortex_x = calculate_order_of_convergence(grid_sizes, vortex_x_values)
    p_vortex_y, ext_vortex_y = calculate_order_of_convergence(grid_sizes, vortex_y_values)
    p_v_center, ext_v_center = calculate_order_of_convergence(grid_sizes, v_center_values)
    
    if Re == 100 and all(x is not None for x in u_rmse_values):
        p_rmse, ext_rmse = calculate_order_of_convergence(grid_sizes, u_rmse_values)
    else:
        p_rmse, ext_rmse = None, None
    
    # Print results table
    print("\nGrid Convergence Study Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Print convergence analysis
    print("\nConvergence Analysis:")
    if p_vortex_x is not None:
        print(f"  Vortex X-position: Order={p_vortex_x:.2f}, Extrapolated value={ext_vortex_x:.6f}")
    if p_vortex_y is not None:
        print(f"  Vortex Y-position: Order={p_vortex_y:.2f}, Extrapolated value={ext_vortex_y:.6f}")
    if p_v_center is not None:
        print(f"  Center V-velocity: Order={p_v_center:.2f}, Extrapolated value={ext_v_center:.6f}")
    if p_rmse is not None:
        print(f"  RMSE against Ghia: Order={p_rmse:.2f}, Extrapolated value={ext_rmse:.6f}")
    
    # Plot grid convergence
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(grid_spacings, vortex_x_values, 'bo-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('Vortex X-position')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(grid_spacings, vortex_y_values, 'ro-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('Vortex Y-position')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(grid_spacings, v_center_values, 'go-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('Center V-velocity')
    plt.grid(True)
    
    if Re == 100 and all(x is not None for x in u_rmse_values):
        plt.subplot(2, 2, 4)
        plt.plot(grid_spacings, u_rmse_values, 'mo-')
        plt.xlabel('Grid Spacing (h)')
        plt.ylabel('RMSE against Ghia')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f"grid_convergence_Re{Re}.png", dpi=300)
    plt.close()
    
    # Plot log-log to verify order
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.loglog(grid_spacings, [abs(x - vortex_x_values[-1]) for x in vortex_x_values], 'bo-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('|Vortex X - Finest Grid Value|')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.loglog(grid_spacings, [abs(y - vortex_y_values[-1]) for y in vortex_y_values], 'ro-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('|Vortex Y - Finest Grid Value|')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.loglog(grid_spacings, [abs(v - v_center_values[-1]) for v in v_center_values], 'go-')
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('|Center V - Finest Grid Value|')
    plt.grid(True)
    
    if Re == 100 and all(x is not None for x in u_rmse_values):
        plt.subplot(2, 2, 4)
        plt.loglog(grid_spacings, u_rmse_values, 'mo-')
        plt.xlabel('Grid Spacing (h)')
        plt.ylabel('RMSE against Ghia')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f"grid_convergence_loglog_Re{Re}.png", dpi=300)
    plt.close()
    
    # Save results to CSV
    with open(output_path / f"grid_convergence_Re{Re}.csv", 'w') as f:
        f.write("Grid,h,Vortex_X,Vortex_Y,Center_V,RMSE\n")
        for i, grid in enumerate(grid_sizes):
            rmse_val = u_rmse_values[i] if u_rmse_values[i] is not None else "N/A"
            f.write(f"{grid},{grid_spacings[i]},{vortex_x_values[i]},{vortex_y_values[i]},{v_center_values[i]},{rmse_val}\n")
    
    # Save convergence analysis
    with open(output_path / f"convergence_analysis_Re{Re}.txt", 'w') as f:
        f.write("Grid Convergence Study Results:\n\n")
        f.write(tabulate(results, headers="keys", tablefmt="grid"))
        f.write("\n\nConvergence Analysis:\n")
        
        if p_vortex_x is not None:
            f.write(f"  Vortex X-position: Order={p_vortex_x:.2f}, Extrapolated value={ext_vortex_x:.6f}\n")
        if p_vortex_y is not None:
            f.write(f"  Vortex Y-position: Order={p_vortex_y:.2f}, Extrapolated value={ext_vortex_y:.6f}\n")
        if p_v_center is not None:
            f.write(f"  Center V-velocity: Order={p_v_center:.2f}, Extrapolated value={ext_v_center:.6f}\n")
        if p_rmse is not None:
            f.write(f"  RMSE against Ghia: Order={p_rmse:.2f}, Extrapolated value={ext_rmse:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid independence study for CFD simulation.")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--grids", type=str, default="17,33,49,65,97", help="Comma-separated list of grid sizes")
    parser.add_argument("--time", type=float, default=20.0, help="Simulation time")
    parser.add_argument("--output", type=str, default="grid_study_results", help="Output directory")
    args = parser.parse_args()
    
    grid_sizes = [int(x) for x in args.grids.split(',')]
    
    run_grid_study(args.re, grid_sizes, args.time, args.output)
