import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

@jit(nopython=True, cache=True)
def _simulation_step(N, dx, dy, dt, nu, u, v, p, p_iterations, alpha_u=0.7, alpha_p=0.3):
    """
    Computes a single time step of the simulation using Chorin's projection method.
    Uses a simpler, more robust implementation for better stability.
    
    Parameters:
    -----------
    alpha_u: float
        Under-relaxation factor for velocity (0 < alpha_u <= 1)
        Lower values increase stability but slow convergence
    alpha_p: float
        Under-relaxation factor for pressure (0 < alpha_p <= 1)
        Lower values increase stability but slow convergence
    """
    un = u.copy()
    vn = v.copy()

    # 1. Predictor step (calculate intermediate velocity u_star, v_star without pressure)
    u_star = un.copy()
    v_star = vn.copy()

    # First-order upwind scheme for convection, central difference for diffusion
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            # CONVECTION TERMS - First-order upwind scheme (robust and stable)
            # u-velocity convection in x-direction
            if un[j, i] > 0:
                du_dx = (un[j, i] - un[j, i-1]) / dx  # backward difference
            else:
                du_dx = (un[j, i+1] - un[j, i]) / dx  # forward difference
            
            # u-velocity convection in y-direction
            if vn[j, i] > 0:
                du_dy = (un[j, i] - un[j-1, i]) / dy  # backward difference
            else:
                du_dy = (un[j+1, i] - un[j, i]) / dy  # forward difference
            
            # v-velocity convection in x-direction
            if un[j, i] > 0:
                dv_dx = (vn[j, i] - vn[j, i-1]) / dx  # backward difference
            else:
                dv_dx = (vn[j, i+1] - vn[j, i]) / dx  # forward difference
            
            # v-velocity convection in y-direction
            if vn[j, i] > 0:
                dv_dy = (vn[j, i] - vn[j-1, i]) / dy  # backward difference
            else:
                dv_dy = (vn[j+1, i] - vn[j, i]) / dy  # forward difference
            
            # DIFFUSION TERMS - Central difference (second-order accurate)
            # u-velocity diffusion
            d2u_dx2 = (un[j, i+1] - 2*un[j, i] + un[j, i-1]) / dx**2
            d2u_dy2 = (un[j+1, i] - 2*un[j, i] + un[j-1, i]) / dy**2
            
            # v-velocity diffusion
            d2v_dx2 = (vn[j, i+1] - 2*vn[j, i] + vn[j, i-1]) / dx**2
            d2v_dy2 = (vn[j+1, i] - 2*vn[j, i] + vn[j-1, i]) / dy**2
            
            # Combine convection and diffusion terms
            u_conv = un[j, i] * du_dx + vn[j, i] * du_dy
            u_diff = nu * (d2u_dx2 + d2u_dy2)
            
            v_conv = un[j, i] * dv_dx + vn[j, i] * dv_dy
            v_diff = nu * (d2v_dx2 + d2v_dy2)
            
            # Update intermediate velocities
            u_star[j, i] = un[j, i] + dt * (-u_conv + u_diff)
            v_star[j, i] = vn[j, i] + dt * (-v_conv + v_diff)

    # Apply velocity boundary conditions to intermediate velocities
    # Bottom wall: no-slip
    u_star[0, :] = 0.0
    v_star[0, :] = 0.0
    
    # Top wall: moving lid
    u_star[-1, 1:-1] = 1.0  # Lid velocity
    u_star[-1, 0] = 0.0     # Corners are no-slip
    u_star[-1, -1] = 0.0    # Corners are no-slip
    v_star[-1, :] = 0.0
    
    # Left and right walls: no-slip
    u_star[:, 0] = 0.0
    u_star[:, -1] = 0.0
    v_star[:, 0] = 0.0
    v_star[:, -1] = 0.0

    # 2. Pressure Poisson Equation
    # Start with zeroed pressure field for stability
    p = np.zeros_like(p)
    
    # Source term for the Poisson equation (divergence of intermediate velocity)
    b = np.zeros((N, N))
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            b[j, i] = (1 / dt) * (
                (u_star[j, i+1] - u_star[j, i-1]) / (2 * dx) +
                (v_star[j+1, i] - v_star[j-1, i]) / (2 * dy)
            )

    # Solve pressure Poisson equation using Jacobi iteration with under-relaxation
    for _ in range(p_iterations):
        p_old = p.copy()
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                p_new = 0.25 * (
                    p_old[j, i+1] + p_old[j, i-1] +
                    p_old[j+1, i] + p_old[j-1, i] -
                    dx**2 * b[j, i]
                )
                
                # Apply under-relaxation for pressure
                p[j, i] = alpha_p * p_new + (1.0 - alpha_p) * p_old[j, i]
        
        # Apply pressure boundary conditions (Neumann)
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = L
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = H (lid)
    
    # Ensure pressure has zero mean (fix reference point)
    p = p - p.mean()

    # 3. Corrector step (update velocities with pressure gradient and apply under-relaxation)
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    # Calculate corrected velocities based on pressure gradient
    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2 * dx)
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2 * dy)
    
    # Apply under-relaxation for velocity update
    u[1:-1, 1:-1] = alpha_u * u_new[1:-1, 1:-1] + (1.0 - alpha_u) * un[1:-1, 1:-1]
    v[1:-1, 1:-1] = alpha_u * v_new[1:-1, 1:-1] + (1.0 - alpha_u) * vn[1:-1, 1:-1]

    # Re-apply velocity boundary conditions
    u[0, :] = 0
    u[-1, 1:-1] = 1.0
    u[:, 0] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    
    return u, v, p

def calculate_cfl(u, v, dx, dy, dt):
    """
    Calculate CFL number based on velocity field and grid parameters
    
    Parameters:
    -----------
    u, v : numpy arrays
        Velocity components
    dx, dy : float
        Grid spacing
    dt : float
        Time step
        
    Returns:
    --------
    cfl_total : numpy array
        CFL values at each grid point
    max_cfl : float
        Maximum CFL value
    """
    # Initialize CFL arrays
    N = u.shape[0]
    cfl_x = np.zeros_like(u)
    cfl_y = np.zeros_like(v)
    
    # Calculate CFL components
    for j in range(1, N-1):
        for i in range(1, N-1):
            cfl_x[j, i] = np.abs(u[j, i]) * dt / dx
            cfl_y[j, i] = np.abs(v[j, i]) * dt / dy
    
    # Total CFL is sum of components
    cfl_total = cfl_x + cfl_y
    max_cfl = np.max(cfl_total)
    
    return cfl_total, max_cfl

def single_step_analysis(Re=100, N=41, dt=0.001, with_plot=True):
    """
    Run one time step of the simulation and analyze CFL condition
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    N : int
        Grid size
    dt : float
        Time step to use
    with_plot : bool
        Whether to generate plots
        
    Returns:
    --------
    cfl_value : float
        Maximum CFL number
    recommended_dt : float
        Recommended time step based on CFL target
    """
    print(f"\nCFL Analysis for Re={Re}")
    print("=" * 50)
    
    # Domain and grid parameters
    L = 1.0
    dx = dy = L / (N - 1)
    nu = 1.0 / Re
    
    # Initialize fields
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    
    # Set lid velocity
    u[-1, 1:-1] = 1.0
    
    # Determine under-relaxation factors based on Re
    if Re <= 100:
        alpha_u, alpha_p = 0.8, 0.5
        p_iterations = 50
    elif Re <= 400:
        alpha_u, alpha_p = 0.7, 0.3
        p_iterations = 100
    else:
        alpha_u, alpha_p = 0.5, 0.2
        p_iterations = 200
    
    print(f"Grid: {N}x{N}, dx={dx:.5f}")
    print(f"Time step: dt={dt:.6f}")
    print(f"Under-relaxation: alpha_u={alpha_u}, alpha_p={alpha_p}")
    
    # Measure execution time
    start_time = time.time()
    
    # Run simulation step
    u, v, p = _simulation_step(N, dx, dy, dt, nu, u, v, p, p_iterations, alpha_u, alpha_p)
    
    elapsed = time.time() - start_time
    print(f"Step execution time: {elapsed:.6f} seconds")
    
    # Calculate CFL
    cfl_total, max_cfl = calculate_cfl(u, v, dx, dy, dt)
    
    # Calculate statistics
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    avg_u = np.mean(np.abs(u))
    avg_v = np.mean(np.abs(v))
    avg_cfl = np.mean(cfl_total)
    
    print(f"\nVelocity statistics:")
    print(f"  Maximum |u|: {max_u:.4f}")
    print(f"  Maximum |v|: {max_v:.4f}")
    print(f"  Average |u|: {avg_u:.4f}")
    print(f"  Average |v|: {avg_v:.4f}")
    
    print(f"\nCFL analysis:")
    print(f"  Maximum CFL: {max_cfl:.4f}")
    print(f"  Average CFL: {avg_cfl:.4f}")
    
    # Calculate recommended time step based on CFL condition
    cfl_target = 0.5  # Conservative for stability
    recommended_dt = cfl_target * dt / max_cfl if max_cfl > 0 else dt
    
    print(f"\nTime step analysis:")
    print(f"  Current dt: {dt:.6f}")
    print(f"  For target CFL={cfl_target}, recommended dt: {recommended_dt:.6f}")
    
    if max_cfl > 1.0:
        print(f"  WARNING: CFL > 1.0, simulation may be unstable")
    elif max_cfl > 0.8:
        print(f"  CAUTION: CFL approaching 1.0, consider reducing time step")
    else:
        print(f"  CFL < 0.8, simulation should be stable")
    
    # Create visualization if requested
    if with_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X, Y = np.meshgrid(x, y)
        
        # Velocity magnitude plot
        im0 = axes[0,0].contourf(X, Y, vel_mag, 20, cmap='viridis')
        axes[0,0].streamplot(X, Y, u, v, color='white', density=1.0, linewidth=0.8)
        axes[0,0].set_title(f'Velocity Magnitude and Streamlines (Re={Re})')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0,0], label='Velocity Magnitude')
        
        # CFL plot
        im1 = axes[0,1].contourf(X, Y, cfl_total, 20, cmap='hot')
        axes[0,1].set_title(f'CFL Distribution (max={max_cfl:.4f})')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,1], label='CFL Value')
        
        # U-velocity plot
        im2 = axes[1,0].contourf(X, Y, u, 20, cmap='coolwarm')
        axes[1,0].set_title('U-Velocity')
        axes[1,0].set_xlabel('X')
        axes[1,0].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1,0], label='U-Velocity')
        
        # V-velocity plot
        im3 = axes[1,1].contourf(X, Y, v, 20, cmap='coolwarm')
        axes[1,1].set_title('V-Velocity')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[1,1], label='V-Velocity')
        
        plt.tight_layout()
        plt.savefig(f'cfl_analysis_detail_re{Re}.png', dpi=300)
        print(f"\nDetailed visualization saved to cfl_analysis_detail_re{Re}.png")
    
    return max_cfl, recommended_dt

if __name__ == "__main__":
    # Run analysis for different Reynolds numbers
    for Re in [100, 400, 1000]:
        # Use a fixed dt for all cases to show the CFL differences
        initial_dt = 0.001
        max_cfl, recommended_dt = single_step_analysis(Re=Re, dt=initial_dt)
        
        # Run a second step with the recommended dt to show improvement
        if max_cfl > 0.8:
            print("\nRunning second step with recommended dt...")
            _, _ = single_step_analysis(Re=Re, dt=recommended_dt)
        
        print("\n" + "=" * 50)
