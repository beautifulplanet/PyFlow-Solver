def quick_face_interpolation_1d(phi, i, flow_positive):
    """
    Compute QUICK face value at the face between i and i+1 in 1D.
    phi: 1D array of cell values
    i: index of the left cell (face is between i and i+1)
    flow_positive: True for positive flow, False for negative
    Returns: interpolated value at the face
    """
    if flow_positive:
        return (3/8)*phi[i-1] + (6/8)*phi[i] - (1/8)*phi[i+1]
    else:
        return (3/8)*phi[i+2] + (6/8)*phi[i+1] - (1/8)*phi[i]
"""
Advection operator module for pyfoamclone CFD solver.

This module will contain implementations of advection (convective) schemes for use in the momentum prediction step.
"""
import numpy as np

def advect_upwind(u, v, field, dx, dy):
    """
    Compute the first-order upwind convective derivative of a field.
    Args:
        u, v: velocity components (2D arrays, shape (ny, nx))
        field: scalar field to be advected (2D array, shape (ny, nx))
        dx, dy: grid spacing
    Returns:
        conv: 2D array, same shape as field, representing u*d(field)/dx + v*d(field)/dy (upwind)
    """
    ny, nx = field.shape
    conv = np.zeros_like(field)
    # x-direction upwind
    for j in range(ny):
        for i in range(nx):
            # Upwind in x
            if u[j, i] > 0:
                fx = (field[j, i] - field[j, i-1]) / dx if i > 0 else (field[j, i] - field[j, i]) / dx
            else:
                fx = (field[j, (i+1) if i+1 < nx else i] - field[j, i]) / dx
            # Upwind in y
            if v[j, i] > 0:
                fy = (field[j, i] - field[j-1, i]) / dy if j > 0 else (field[j, i] - field[j, i]) / dy
            else:
                fy = (field[(j+1) if j+1 < ny else j, i] - field[j, i]) / dy
            conv[j, i] = u[j, i] * fx + v[j, i] * fy
    return conv

def advect_quick(u, v, field, dx, dy):
    """
    Compute the convective derivative using the QUICK (Quadratic Upstream Interpolation for Convective Kinematics) scheme.
    Third-order accurate for uniform grids. Uses a wider stencil than upwind.
    Args:
        u, v: velocity components (2D arrays, shape (ny, nx))
        field: scalar field to be advected (2D array, shape (ny, nx))
        dx, dy: grid spacing
    Returns:
        conv: 2D array, same shape as field, representing u*d(field)/dx + v*d(field)/dy (QUICK)
    """
    ny, nx = field.shape
    conv = np.zeros_like(field)
    phi = np.pad(field, 2, mode='edge')
    for j in range(ny):
        for i in range(nx):
            # --- X-Fluxes for cell (j, i) ---
            if u[j, i] >= 0:
                phi_e = (3/8)*phi[j+2, i+3] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+2, i+1]
            else:
                phi_e = (3/8)*phi[j+2, i+1] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+2, i+3]
            if u[j, i] >= 0:
                phi_w = (3/8)*phi[j+2, i+2] + (6/8)*phi[j+2, i+1] - (1/8)*phi[j+2, i]
            else:
                phi_w = (3/8)*phi[j+2, i] + (6/8)*phi[j+2, i+1] - (1/8)*phi[j+2, i+2]
            # --- Y-Fluxes for cell (j, i) ---
            if v[j, i] >= 0:
                phi_n = (3/8)*phi[j+3, i+2] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+1, i+2]
            else:
                phi_n = (3/8)*phi[j+1, i+2] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+3, i+2]
            if v[j, i] >= 0:
                phi_s = (3/8)*phi[j+2, i+2] + (6/8)*phi[j+1, i+2] - (1/8)*phi[j, i+2]
            else:
                phi_s = (3/8)*phi[j, i+2] + (6/8)*phi[j+1, i+2] - (1/8)*phi[j+2, i+2]
            flux_x_e = u[j, i] * phi_e
            flux_x_w = u[j, i] * phi_w
            flux_y_n = v[j, i] * phi_n
            flux_y_s = v[j, i] * phi_s
            conv[j, i] = (flux_x_e - flux_x_w) / dx + (flux_y_n - flux_y_s) / dy
    return conv
    """
    Compute the first-order upwind convective derivative of a field.
    Args:
        u, v: velocity components (2D arrays, shape (ny, nx))
        field: scalar field to be advected (2D array, shape (ny, nx))
        dx, dy: grid spacing
    Returns:
        conv: 2D array, same shape as field, representing u*d(field)/dx + v*d(field)/dy (upwind)
    """
    ny, nx = field.shape
    conv = np.zeros_like(field)
    # x-direction upwind
    for j in range(ny):
        for i in range(nx):
            # Upwind in x
            if u[j, i] > 0:
                fx = (field[j, i] - field[j, i-1]) / dx if i > 0 else (field[j, i] - field[j, i]) / dx
            else:
                fx = (field[j, (i+1) if i+1 < nx else i] - field[j, i]) / dx
            # Upwind in y
            if v[j, i] > 0:
                fy = (field[j, i] - field[j-1, i]) / dy if j > 0 else (field[j, i] - field[j, i]) / dy
            else:
                fy = (field[(j+1) if j+1 < ny else j, i] - field[j, i]) / dy
            conv[j, i] = u[j, i] * fx + v[j, i] * fy
    return conv
