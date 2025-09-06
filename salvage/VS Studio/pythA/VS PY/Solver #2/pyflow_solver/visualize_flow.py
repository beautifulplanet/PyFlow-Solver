import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from pyflow.solver import solve_lid_driven_cavity
from pyflow.grid import Grid

def run_simulation(Re, N=41, T=5.0):
    """
    Run a simulation for the specified Reynolds number
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    N : int
        Grid size
    T : float
        Simulation time
    
    Returns:
    --------
    u, v, p : numpy arrays
        Velocity and pressure fields
    grid : Grid
        Grid object
    residuals : dict
        Dictionary containing residual histories
    computation_time : float
        Total computation time in seconds
    """
    # Set up grid
    L = 1.0
    grid = Grid(N, L)
    dx, dy = grid.dx, grid.dy
    
    # Set initial time step
    dt_init = 0.001
    
    # Run simulation with timing
    start_time = time.time()
    print(f"Processing Re={Re}...")
    
    # Use CFL-based adaptive time stepping for stability
    if Re <= 100:
        cfl_target = 0.5
        max_dt = 0.0025
    elif Re <= 400:
        cfl_target = 0.5
        max_dt = 0.001
    else:
        cfl_target = 0.5
        max_dt = 0.0005
    
    # Run the simulation
    u, v, p, residuals = solve_lid_driven_cavity(
        N=N,
        dx=dx,
        dy=dy,
        Re=Re,
        dt=dt_init,
        T=T,
        use_cfl=True,
        cfl_target=cfl_target
    )
    
    # Calculate computation time
    computation_time = time.time() - start_time
    
    # Count timesteps
    n_steps = len(residuals['u_res'])
    
    print(f"Simulation at Re={Re} complete. ")
    print(f"Total time steps: {n_steps}")
    print(f"Simulation time: {T} seconds")
    print(f"Average computational time per timestep: {computation_time/n_steps:.4f} seconds")
    print(f"Overall computation time: {computation_time:.2f} seconds")
    print()
    
    return u, v, p, grid, residuals, computation_time

def plot_velocity_magnitude(u, v, grid, Re, ax=None, title=None, with_streamlines=True):
    """
    Plot velocity magnitude with optional streamlines
    
    Parameters:
    -----------
    u, v : numpy arrays
        Velocity field components
    grid : Grid object
        Computational grid
    Re : float
        Reynolds number for the title
    ax : matplotlib axis, optional
        Axis to plot on
    title : str, optional
        Title for the plot
    with_streamlines : bool
        Whether to add streamlines to the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create mesh grid for plotting
    x = np.linspace(0, grid.Lx, grid.Nx)
    y = np.linspace(0, grid.Ly, grid.Ny)
    X, Y = np.meshgrid(x, y)
    
    # Create a custom colormap for velocity magnitude
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1), 
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('velocity', colors, N=100)
    
    # Plot velocity magnitude
    cf = ax.contourf(X, Y, vel_mag, 20, cmap=cmap)
    plt.colorbar(cf, ax=ax, label='Velocity Magnitude')
    
    # Add streamlines if requested
    if with_streamlines:
        # Reduce density for clearer visualization
        step = max(1, grid.Nx // 20)
        ax.streamplot(X, Y, u, v, color='white', density=1.0, 
                      linewidth=0.8, arrowsize=1)
    
    # Set the title
    if title is None:
        title = f'Velocity Magnitude for Re={Re}'
    ax.set_title(title)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return ax

def plot_residuals(residuals, Re, ax=None):
    """
    Plot convergence history of residuals
    
    Parameters:
    -----------
    residuals : dict
        Dictionary with keys 'u_res', 'v_res', 'cont_res'
    Re : float
        Reynolds number for the title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot residuals
    iterations = np.arange(1, len(residuals['u_res'])+1)
    ax.semilogy(iterations, residuals['u_res'], 'b-', label='u-velocity')
    ax.semilogy(iterations, residuals['v_res'], 'r-', label='v-velocity')
    ax.semilogy(iterations, residuals['cont_res'], 'g-', label='continuity')
    
    # Add labels and title
    ax.set_title(f'Convergence History for Re={Re}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    
    return ax

def validate_centerline_velocity(u, grid, Re):
    """
    Compare centerline velocity with Ghia et al. (1982) benchmark data
    
    Parameters:
    -----------
    u : numpy array
        x-velocity field
    grid : Grid object
        Computational grid
    Re : float
        Reynolds number
    
    Returns:
    --------
    fig : matplotlib figure
        Figure with validation plot
    """
    # Get vertical centerline u-velocity
    mid_x = grid.Nx // 2
    centerline_u = u[:, mid_x]
    y_coords = np.linspace(0, 1, grid.Ny)
    
    # Ghia et al. (1982) data for Re=100
    if Re == 100:
        ghia_y = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 
                           0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
        ghia_u = np.array([0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, 
                           -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 
                           0.78871, 0.84123, 1.0])
    elif Re == 400:
        ghia_y = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5,
                          0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
        ghia_u = np.array([0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, 
                          -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 
                          0.68439, 0.75837, 1.0])
    elif Re == 1000:
        ghia_y = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5,
                          0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
        ghia_u = np.array([0, -0.18109, -0.20196, -0.22220, -0.29012, -0.38289, -0.27805, 
                          -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 
                          0.57492, 0.65928, 1.0])
    else:
        # No benchmark data for other Reynolds numbers
        print(f"No benchmark data available for Re={Re}")
        return None
    
    # Create validation plot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(centerline_u, y_coords, 'b-', label='Our Solution')
    ax.plot(ghia_u, ghia_y, 'ro', label='Ghia et al. (1982)')
    ax.set_title(f'Centerline u-velocity Validation, Re={Re}')
    ax.set_xlabel('u-velocity')
    ax.set_ylabel('y-coordinate')
    ax.grid(True)
    ax.legend()
    
    return fig

if __name__ == "__main__":
    # Run simulations for different Reynolds numbers
    reynolds_numbers = [100, 400, 1000]
    results = {}
    
    # Set up figure for velocity magnitude plots
    fig_vel, axes_vel = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set up figure for residual plots
    fig_res, axes_res = plt.subplots(1, 3, figsize=(15, 5))
    
    # Run simulations and plot results
    for i, Re in enumerate(reynolds_numbers):
        u, v, p, grid, residuals, computation_time = run_simulation(Re)
        results[Re] = {'u': u, 'v': v, 'p': p, 'grid': grid, 
                       'residuals': residuals, 'time': computation_time}
        
        # Plot velocity magnitude
        plot_velocity_magnitude(u, v, grid, Re, ax=axes_vel[i])
        
        # Plot residuals
        plot_residuals(residuals, Re, ax=axes_res[i])
    
    fig_vel.tight_layout()
    fig_vel.savefig('velocity_fields.png', dpi=300)
    
    fig_res.tight_layout()
    fig_res.savefig('residuals.png', dpi=300)
    
    # Create validation plot for Re=100
    print("Creating validation plot for Re=100...")
    Re = 100
    fig_val = validate_centerline_velocity(results[Re]['u'], results[Re]['grid'], Re)
    fig_val.savefig('validation_re100.png', dpi=300)
    print("Benchmark validation completed.")
    
    print("\nSaving visualization plots")
    
    plt.close('all')
