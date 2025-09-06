"""
This module provides a hybrid Python/C++ implementation of the CFD solver.
It integrates the high-performance C++ functions with the Python solver.
"""

import numpy as np
from .grid import Grid
import os
import sys

# Try to import the C++ module
try:
    import pyflow_core_cfd
    _HAVE_CPP_EXTENSION = True
    print("Using C++ extension for performance-critical functions")
except ImportError:
    _HAVE_CPP_EXTENSION = False
    import warnings
    warnings.warn("C++ extension 'pyflow_core_cfd' not available, using pure Python implementation")
    import numpy as np  # Ensure np is always available

def solve_pressure_poisson(b, p_init, dx, dy, max_iter=1000, tol=1e-4, alpha_p=0.8):
    """
    Solve the pressure Poisson equation using Jacobi iteration.
    This function will use the C++ implementation if available, otherwise fallback to Python.
    
    Parameters:
    ----------
    b : ndarray
        Source term (right-hand side) of the Poisson equation
    p_init : ndarray
        Initial guess for the pressure field
    dx, dy : float
        Grid spacing in x and y directions
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    alpha_p : float
        Under-relaxation factor for pressure
        
    Returns:
    -------
    ndarray
        The solution of the pressure field
    """
    if _HAVE_CPP_EXTENSION:
        # Use the C++ implementation
        return pyflow_core_cfd.solve_pressure_poisson(
            b, p_init, dx, dy, max_iter, tol, alpha_p
        )
    else:
        # Fallback to a Python implementation
        p = p_init.copy()
        dx2 = dx * dx
        dy2 = dy * dy
        coef = 2.0 * (1.0/dx2 + 1.0/dy2)
        error = 1.0
        iter_count = 0
        ny, nx = b.shape
        p_temp = np.zeros_like(p)
        while error > tol and iter_count < max_iter:
            # Jacobi iteration: update p_temp from p
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    p_new = (
                        (p[i, j+1] + p[i, j-1]) / dx2 +
                        (p[i+1, j] + p[i-1, j]) / dy2 -
                        b[i, j]
                    ) / coef
                    p_temp[i, j] = p[i, j] + alpha_p * (p_new - p[i, j])
            # Apply boundary conditions to p_temp
            p_temp[0, :] = p_temp[1, :]        # bottom
            p_temp[-1, :] = p_temp[-2, :]      # top
            p_temp[:, 0] = p_temp[:, 1]        # left
            p_temp[:, -1] = p_temp[:, -2]      # right
            # Calculate error before swapping
            error = np.sum(np.abs(p_temp - p)) / (nx * ny)
            # Swap p and p_temp for next iteration
            p, p_temp = p_temp, p
            iter_count += 1
        print(f"Pressure solver converged in {iter_count} iterations with error {error}")
        return p

def calculate_pressure_source(u, v, dx, dy, dt):
    """
    Calculate the source term for the pressure equation.
    
    Parameters:
    ----------
    u, v : ndarray
        Velocity components
    dx, dy : float
        Grid spacing
    dt : float
        Time step
        
    Returns:
    -------
    ndarray
        The source term for the pressure equation
    """
    if _HAVE_CPP_EXTENSION:
        # Use the C++ implementation
        return pyflow_core_cfd.calculate_pressure_source(u, v, dx, dy, dt)
    else:
        # Fallback to a Python implementation
        ny, nx = u.shape
        b = np.zeros_like(u)
        
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                du_dx = (u[i, j+1] - u[i, j-1]) / (2.0 * dx)
                dv_dy = (v[i+1, j] - v[i-1, j]) / (2.0 * dy)
                
                b[i, j] = -1.0/dt * (du_dx + dv_dy)
        
        # Set boundary values to zero
        b[0, :] = b[-1, :] = 0.0
        b[:, 0] = b[:, -1] = 0.0
        
        return b

def correct_velocities(u, v, p, dx, dy, dt):
    """
    Correct velocities based on the pressure gradient to enforce continuity.
    
    Parameters:
    ----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    dx, dy : float
        Grid spacing
    dt : float
        Time step
        
    Returns:
    -------
    tuple
        Corrected velocity components (u, v)
    """
    if _HAVE_CPP_EXTENSION:
        # Use the C++ implementation
        return pyflow_core_cfd.correct_velocities(u, v, p, dx, dy, dt)
    else:
        # Fallback to a Python implementation
        ny, nx = u.shape
        u_corr = u.copy()
        v_corr = v.copy()
        
        # Correct u-velocity
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                dp_dx = (p[i, j+1] - p[i, j-1]) / (2.0 * dx)
                u_corr[i, j] = u[i, j] - dt * dp_dx
        
        # Correct v-velocity
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                dp_dy = (p[i+1, j] - p[i-1, j]) / (2.0 * dy)
                v_corr[i, j] = v[i, j] - dt * dp_dy
        
        # Apply boundary conditions
        # Bottom and top (no-slip)
        u_corr[0, :] = 0.0
        u_corr[-1, :] = 1.0  # Lid velocity
        v_corr[0, :] = 0.0
        v_corr[-1, :] = 0.0
        
        # Left and right (no-slip)
        u_corr[:, 0] = u_corr[:, -1] = 0.0
        v_corr[:, 0] = v_corr[:, -1] = 0.0
        
        return u_corr, v_corr

def calculate_residuals(u, v, u_prev, v_prev, p):
    """
    Calculate the residuals of the solution.
    
    Parameters:
    ----------
    u, v : ndarray
        Current velocity components
    u_prev, v_prev : ndarray
        Previous velocity components
    p : ndarray
        Pressure field
        
    Returns:
    -------
    dict
        Dictionary containing the residuals for u, v, and continuity
    """
    if _HAVE_CPP_EXTENSION:
        # Use the C++ implementation
        return pyflow_core_cfd.calculate_residuals(u, v, u_prev, v_prev, p)
    else:
        # Fallback to a Python implementation
        ny, nx = u.shape
        
        # Calculate momentum residuals
        u_res = np.sum(np.abs(u[1:-1, 1:-1] - u_prev[1:-1, 1:-1])) / ((ny-2) * (nx-2))
        v_res = np.sum(np.abs(v[1:-1, 1:-1] - v_prev[1:-1, 1:-1])) / ((ny-2) * (nx-2))
        
        # Calculate continuity residual
        cont_res = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                div = (u[i, j+1] - u[i, j-1]) + (v[i+1, j] - v[i-1, j])
                cont_res += np.abs(div)
        
        cont_res /= ((ny-2) * (nx-2) * 2.0)
        
        return {
            'u_res': u_res,
            'v_res': v_res,
            'cont_res': cont_res
        }

def solve_lid_driven_cavity(NPOINTS, dx, dy, Re, dt, T, p_iterations=500, alpha_u=0.8, alpha_p=0.5, logger=None):
    """
    Solve the lid-driven cavity problem using a hybrid Python/C++ implementation.
    
    Parameters:
    ----------
    NPOINTS : int
        Number of grid points in each direction
    dx, dy : float
        Grid spacing
    Re : float
        Reynolds number
    dt : float
        Time step size
    T : float
        Total simulation time
    p_iterations : int
        Maximum number of iterations for pressure solver
    alpha_u, alpha_p : float
        Under-relaxation factors for velocity and pressure
    logger : object
        Logger object for recording results
        
    Returns:
    -------
    tuple
        u, v velocity components, pressure field, and residuals
    """
    # Initialize variables
    nt = int(T / dt)  # Number of time steps
    
    # Create the grid and initialize fields
    grid = Grid(NPOINTS, 1.0)
    
    u = np.zeros((NPOINTS, NPOINTS))  # x-velocity
    v = np.zeros((NPOINTS, NPOINTS))  # y-velocity
    p = np.zeros((NPOINTS, NPOINTS))  # pressure
    
    # Set lid velocity (top boundary)
    u[-1, 1:-1] = 1.0
    
    # Store residuals
    residuals = {
        'u_res': [],
        'v_res': [],
        'cont_res': []
    }
    
    print(f"Using under-relaxation factors: alpha_u={alpha_u}, alpha_p={alpha_p}")
    print(f"Starting simulation:")
    print(f"  Grid size (N): {NPOINTS}")
    print(f"  Reynolds (Re): {Re}")
    print(f"  Time step (dt): {dt}")
    print(f"  Total time (T): {T}")
    print(f"  Total steps: {nt}")
    print("  Logging every: {} steps".format(logger.log_interval if logger else "N/A"))
    print("-" * 60)
    
    # Main time-stepping loop
    u_res = v_res = cont_res = 0.0
    for n in range(1, nt + 1):
        # Store previous velocity for residual calculation
        u_prev = u.copy()
        v_prev = v.copy()
        # Calculate pressure source term
        b = calculate_pressure_source(u, v, dx, dy, dt)
        # Solve pressure Poisson equation
        p = solve_pressure_poisson(
            b, p, dx, dy, 
            max_iter=p_iterations, 
            tol=1e-4, 
            alpha_p=alpha_p
        )
        # Correct velocities to enforce continuity
        u_star, v_star = correct_velocities(u, v, p, dx, dy, dt)
        # Apply under-relaxation for stability
        u = u_prev + alpha_u * (u_star - u_prev)
        v = v_prev + alpha_u * (v_star - v_prev)
        # Calculate residuals
        res = calculate_residuals(u, v, u_prev, v_prev, p)
        u_res = res['u_res']
        v_res = res['v_res']
        cont_res = res['cont_res']
        residuals['u_res'].append(u_res)
        residuals['v_res'].append(v_res)
        residuals['cont_res'].append(cont_res)
        # Only check for nan if all are not None
        if u_res is not None and v_res is not None and cont_res is not None:
            import numpy as np
            if np.isnan(u_res) or np.isnan(v_res) or np.isnan(cont_res):
                raise ValueError("Divergence detected: NaN in residuals.")
        # Log progress if logger is provided
        if logger and n % logger.log_interval == 0:
            logger.log_step(n, u, v, p, res)
    return u, v, p, residuals
