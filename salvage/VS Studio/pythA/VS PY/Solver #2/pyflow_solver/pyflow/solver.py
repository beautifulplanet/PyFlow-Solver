import numpy as np
from typing import Tuple, Callable, Optional
from numba import jit
from .logging import LiveLogger

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
    # Source term for the Poisson equation (divergence of intermediate velocity)
    b = np.zeros((N, N))
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            # Use direct differencing for better mass conservation
            b[j, i] = (1 / dt) * (
                (u_star[j, i] - u_star[j, i-1]) / dx +
                (v_star[j, i] - v_star[j-1, i]) / dy
            )
    
    # Initialize pressure correction field
    p_prime = np.zeros_like(p)
    
    # Solve pressure Poisson equation using Jacobi iteration with under-relaxation
    for iter_count in range(p_iterations):
        p_old = p_prime.copy()
        residual = 0.0
        
        # Update pressure field
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                rhs = b[j, i]
                p_new = 0.25 * (
                    p_old[j, i+1] + p_old[j, i-1] +
                    p_old[j+1, i] + p_old[j-1, i] -
                    dx*dy * rhs
                )
                
                # Calculate residual for convergence check
                res = p_new - p_old[j, i]
                residual += res * res
                
                # Apply under-relaxation for pressure
                p_prime[j, i] = alpha_p * p_new + (1.0 - alpha_p) * p_old[j, i]
        
        # Apply pressure boundary conditions (Neumann)
        p_prime[:, -1] = p_prime[:, -2]  # dp/dx = 0 at x = L
        p_prime[:, 0] = p_prime[:, 1]    # dp/dx = 0 at x = 0
        p_prime[0, :] = p_prime[1, :]    # dp/dy = 0 at y = 0
        p_prime[-1, :] = p_prime[-2, :]  # dp/dy = 0 at y = H (lid)
        
        # Check for early convergence 
        if iter_count % 50 == 0 and iter_count > 0:
            # L2-norm of residual
            residual = np.sqrt(residual) / (N*N)
            if residual < 1e-5:
                break
    
    # Set a reference point for pressure (corner)
    p_prime = p_prime - p_prime[1, 1]
    
    # Update the pressure field with the correction
    p = p + p_prime

    # 3. Corrector step (update velocities with pressure gradient and apply under-relaxation)
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    # Calculate corrected velocities based on pressure gradient
    # Use face-centered pressure gradients for better mass conservation
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            # u-velocity correction at i+1/2,j (cell face)
            if i < N-2:  # Not at right boundary
                dp_dx = (p[j, i+1] - p[j, i]) / dx
                u_new[j, i] = u_star[j, i] - dt * dp_dx
            
            # v-velocity correction at i,j+1/2 (cell face)
            if j < N-2:  # Not at top boundary
                dp_dy = (p[j+1, i] - p[j, i]) / dy
                v_new[j, i] = v_star[j, i] - dt * dp_dy
    
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

def solve_lid_driven_cavity(
    N: int, 
    dx: float, 
    dy: float, 
    Re: float, 
    dt: float, 
    T: float, 
    p_iterations: int = 2000,
    alpha_u: float = None,
    alpha_p: float = None,
    use_cfl: bool = True,
    cfl_target: float = 0.8,
    logger: Optional[LiveLogger] = None
) -> tuple:
    """
    Wrapper for the 2D lid-driven cavity solver that provides real-time progress updates.
    Handles higher Reynolds numbers with adaptive time-stepping and improved stability.
    
    Parameters:
    -----------
    alpha_u: float, optional
        Under-relaxation factor for velocity (0 < alpha_u <= 1)
        If None, adaptive values based on Reynolds number are used
    alpha_p: float, optional
        Under-relaxation factor for pressure (0 < alpha_p <= 1)
        If None, adaptive values based on Reynolds number are used
    use_cfl: bool, optional
        Whether to use CFL-based adaptive time stepping (default: True)
    cfl_target: float, optional
        Target CFL number for adaptive time stepping (default: 0.8)
    """
    if Re <= 0:
        raise ValueError("Reynolds number must be positive.")
    
    # Set initial conservative time step based on Reynolds number
    initial_dt = dt
    max_dt = dt  # Maximum allowed time step
    
    # For higher Reynolds, we need smaller initial time steps
    if Re > 500 and max_dt > 0.0005:
        print(f"Warning: Reducing maximum timestep from {max_dt} to 0.0005 for stability at Re={Re}")
        max_dt = 0.0005
    elif Re > 1000 and max_dt > 0.0001:
        print(f"Warning: Reducing maximum timestep from {max_dt} to 0.0001 for stability at Re={Re}")
        max_dt = 0.0001
        
    # Adjust pressure iterations based on Reynolds number
    # Higher Reynolds requires more pressure iterations
    adaptive_p_iterations = p_iterations
    if Re >= 400 and p_iterations < 5000:
        adaptive_p_iterations = 5000
    if Re >= 1000 and p_iterations < 10000:
        adaptive_p_iterations = 10000
    
    # Set adaptive under-relaxation factors based on Reynolds number if not provided
    if alpha_u is None:
        if Re <= 100:
            alpha_u = 0.8  # Less aggressive for lower Re
        elif Re <= 400:
            alpha_u = 0.7  # Moderate for medium Re
        else:
            alpha_u = 0.5  # Conservative for high Re
    
    if alpha_p is None:
        if Re <= 100:
            alpha_p = 0.5  # Less aggressive for lower Re
        elif Re <= 400:
            alpha_p = 0.3  # Moderate for medium Re
        else:
            alpha_p = 0.2  # Conservative for high Re
    
    print(f"Using under-relaxation factors: alpha_u={alpha_u}, alpha_p={alpha_p}")

    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    
    # Set initial boundary conditions
    u[-1, 1:-1] = 1.0 # Lid velocity

    nu = 1.0 / Re
    current_time = 0.0
    current_dt = initial_dt
    step_count = 0
    
    # Store residuals for monitoring convergence
    u_residuals = []
    v_residuals = []
    cont_residuals = []

    if logger:
        logger.log_header()

    # Initialize residuals to safe values
    u_res = v_res = cont_res = 0.0
    # Run simulation until target time is reached
    while current_time < T:
        # Calculate adaptive time step based on CFL if enabled
        if use_cfl and step_count > 0:  # Skip first step since velocities are initially zero
            u_max = np.max(np.abs(u))
            v_max = np.max(np.abs(v))
            
            # Avoid division by zero
            if u_max > 1e-10 or v_max > 1e-10:
                dt_cfl = cfl_target / (u_max/dx + v_max/dy + 1e-10)
                current_dt = min(dt_cfl, max_dt)  # Limit to max_dt for stability
            else:
                current_dt = max_dt
        
        # Make sure we don't overshoot the final time
        if current_time + current_dt > T:
            current_dt = T - current_time
        
        # Store previous state for residual calculation
        u_old = u.copy()
        v_old = v.copy()
        
        # Run simulation step
        u, v, p = _simulation_step(N, dx, dy, current_dt, nu, u, v, p, adaptive_p_iterations, alpha_u, alpha_p)
        
        # Calculate residuals (L2 norm of change)
        if step_count > 0:  # Skip first step
            u_res = v_res = cont_res = np.nan
            # For test runs, only calculate residuals occasionally to save time
            if step_count % 10 == 0 or current_time + current_dt >= T:
                # Velocity change residuals
                u_res = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u**2) + 1e-10)
                v_res = np.sqrt(np.sum((v - v_old)**2)) / np.sqrt(np.sum(v**2) + 1e-10)
                # Simplified continuity residual calculation (sample only part of the domain)
                div_sum = 0.0
                sample_count = 0
                step_size = max(1, (N-2) // 5)  # Sample roughly 1/5th of the domain
                for j in range(1, N-1, step_size):
                    for i in range(1, N-1, step_size):
                        # Use direct differencing for consistency with pressure equation
                        div = (u[j,i] - u[j,i-1])/dx + (v[j,i] - v[j-1,i])/dy
                        div_sum += abs(div)
                        sample_count += 1
                # Normalize by domain size for more consistent scaling
                cont_res = div_sum / sample_count if sample_count > 0 else 0.0
                # Store residuals
                u_residuals.append(u_res)
                v_residuals.append(v_res)
                cont_residuals.append(cont_res)
                # Check for divergence only if residuals were calculated
                if np.isnan(u_res) or np.isnan(v_res) or np.isnan(cont_res):
                    print(f"DIVERGENCE DETECTED: NaN residuals at step {step_count}")
                    break
            else:
                # For skipped steps, duplicate the last residual to maintain array length
                if len(u_residuals) > 0:
                    u_residuals.append(u_residuals[-1])
                    v_residuals.append(v_residuals[-1]) 
                    cont_residuals.append(cont_residuals[-1])
        
        # Update time and step count
        current_time += current_dt
        step_count += 1
        
        # Check for instability periodically (less frequently for tests)
        # For small grids, check every 200 steps to reduce overhead
        check_interval = 200 if N < 50 else 100
        if step_count % check_interval == 0:
            # Quick check for NaN values which indicate divergence
            if np.isnan(np.sum(u)) or np.isnan(np.sum(v)) or np.isnan(np.sum(p)):
                print(f"DIVERGENCE DETECTED: NaN values found at step {step_count}")
                break
            
            # More thorough check only occasionally
            if step_count % (check_interval * 5) == 0:
                maxu, maxv, maxp = np.nanmax(np.abs(u)), np.nanmax(np.abs(v)), np.nanmax(np.abs(p))
                if maxu > 5 or maxv > 5 or maxp > 50:  # Lower thresholds for early detection
                    print(f"UNPHYSICAL GROWTH DETECTED: max|u|={maxu}, max|v|={maxv}, max|p|={maxp}")
                    break
            
            # Print progress with adaptive dt and residuals information
            if len(u_residuals) > 0:
                print(f"Time: {current_time:.4f}/{T:.4f} | dt: {current_dt:.6f} | " +
                      f"Residuals: u={u_residuals[-1]:.2e}, v={v_residuals[-1]:.2e}, cont={cont_residuals[-1]:.2e}")
        
        if logger:
            logger.log_step(step_count, u, v, p)

    if logger:
        logger.log_footer()

    # Final stability check
    maxu, maxv, maxp = np.nanmax(np.abs(u)), np.nanmax(np.abs(v)), np.nanmax(np.abs(p))
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(p))):
        print(f"DIVERGENCE DETECTED: max|u|={maxu}, max|v|={maxv}, max|p|={maxp}")
    elif maxu > 5 or maxv > 5 or maxp > 50:
        print(f"UNPHYSICAL GROWTH DETECTED: max|u|={maxu}, max|v|={maxv}, max|p|={maxp}")

    # Check final continuity/mass conservation
    div_final = 0.0
    for j in range(1, N-1):
        for i in range(1, N-1):
            div_cell = (u[j,i] - u[j,i-1])/dx + (v[j,i] - v[j-1,i])/dy
            div_final += abs(div_cell)
    div_final /= ((N-2)*(N-2))
    
    # Print final statistics
    if len(u_residuals) > 0:
        print(f"Final residuals: u={u_residuals[-1]:.2e}, v={v_residuals[-1]:.2e}, cont={cont_residuals[-1]:.2e}")
        print(f"Final continuity error: {div_final:.2e}")
        print(f"Total steps: {step_count}, Final time: {current_time}")
    
    # Return solution and residuals
    result_data = {'u_res': u_residuals, 'v_res': v_residuals, 'cont_res': cont_residuals}
    return (u, v, p, result_data)
