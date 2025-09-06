import numpy as np

def advect_quick_reference(field, u, v, dx, dy):
    """
    Minimal, reference-style QUICK advection operator for 2D uniform grids.
    Assumes u, v, field are 2D arrays of the same shape.
    Applies QUICK in the interior, upwind at boundaries.
    Returns: conv (2D array, same shape as field)
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
