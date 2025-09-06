import numpy as np
import matplotlib.pyplot as plt
from numba import jit

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

def run_single_step_with_cfl_analysis(Re=100, N=41):
    """
    Run a single simulation step and calculate the CFL number
    
    Parameters:
    -----------
    Re: float
        Reynolds number
    N: int
        Grid size
    
    Returns:
    --------
    cfl_value: float
        Maximum CFL number after the step
    u, v, p: numpy arrays
        Velocity and pressure fields after the step
    """
    print(f"Running single step CFL analysis with Re={Re}, grid={N}x{N}")
    
    # Setup parameters
    L = 1.0  # Domain size (square)
    dx = dy = L / (N - 1)  # Grid spacing
    nu = 1.0 / Re  # Kinematic viscosity
    
    # Initial time step - will be adjusted by CFL later
    dt_init = 0.001
    
    # Initialize fields
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    
    # Set lid velocity (top boundary)
    u[-1, 1:-1] = 1.0
    
    # Under-relaxation factors based on Reynolds number
    if Re <= 100:
        alpha_u, alpha_p = 0.8, 0.5
    elif Re <= 400:
        alpha_u, alpha_p = 0.7, 0.3
    else:
        alpha_u, alpha_p = 0.5, 0.2
    
    print(f"Using under-relaxation factors: alpha_u={alpha_u}, alpha_p={alpha_p}")
    
    # Pressure iterations
    p_iterations = 50
    if Re >= 400:
        p_iterations = 100
    if Re >= 1000:
        p_iterations = 200
    
    # Run one time step
    u, v, p = _simulation_step(N, dx, dy, dt_init, nu, u, v, p, p_iterations, alpha_u, alpha_p)
    
    # Calculate CFL number
    cfl_x = np.zeros_like(u)
    cfl_y = np.zeros_like(v)
    
    # Calculate local CFL numbers
    for j in range(1, N-1):
        for i in range(1, N-1):
            cfl_x[j, i] = np.abs(u[j, i]) * dt_init / dx
            cfl_y[j, i] = np.abs(v[j, i]) * dt_init / dy
    
    # Total CFL is the sum of components
    cfl_total = cfl_x + cfl_y
    max_cfl = np.max(cfl_total)
    max_u = np.max(np.abs(u))
    max_v = np.max(np.abs(v))
    
    print(f"Maximum velocity: |u|_max = {max_u:.4f}, |v|_max = {max_v:.4f}")
    print(f"Maximum CFL number: {max_cfl:.4f}")
    print(f"For stability, CFL should be < 1.0")
    
    # Calculate adaptive time step based on this CFL
    cfl_target = 0.5  # Conservative target
    dt_cfl = cfl_target / (max_u / dx + max_v / dy) if (max_u / dx + max_v / dy) > 0 else dt_init
    print(f"Recommended dt for next step: {dt_cfl:.6f}")
    
    # Create visualization of velocity field
    plt.figure(figsize=(10, 8))
    
    # Plot velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create a mesh grid for plotting
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # Plot velocity magnitude with contours
    plt.contourf(X, Y, vel_mag, 20, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude')
    
    # Add streamlines to visualize flow pattern
    plt.streamplot(X, Y, u, v, color='white', density=1.0)
    
    # Add CFL contours
    cfl_contour = plt.contour(X, Y, cfl_total, levels=[0.1, 0.2, 0.3, 0.4, 0.5], 
                             colors='red', linewidths=1)
    plt.clabel(cfl_contour, inline=True, fontsize=8, fmt='%.1f')
    
    plt.title(f'Velocity Field and CFL Contours (Re={Re})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'cfl_analysis_re{Re}.png', dpi=300)
    
    return max_cfl, u, v, p

# Example usage
if __name__ == "__main__":
    # Run for different Reynolds numbers to see the effect on CFL
    for Re in [100, 400, 1000]:
        cfl, u, v, p = run_single_step_with_cfl_analysis(Re=Re)
        print("-" * 50)
