import numpy as np

def advect_upwind(u, v, field, dx, dy):
    ny, nx = field.shape
    conv = np.zeros_like(field)
    for j in range(ny):
        for i in range(nx):
            if u[j, i] > 0:
                fx = (field[j, i] - field[j, i-1]) / dx if i > 0 else 0.0
            else:
                fx = (field[j, (i+1) if i+1 < nx else i] - field[j, i]) / dx
            if v[j, i] > 0:
                fy = (field[j, i] - field[j-1, i]) / dy if j > 0 else 0.0
            else:
                fy = (field[(j+1) if j+1 < ny else j, i] - field[j, i]) / dy
            conv[j, i] = u[j, i] * fx + v[j, i] * fy
    return conv

def advect_quick(u, v, field, dx, dy):
    ny, nx = field.shape
    conv = np.zeros_like(field)
    # QUICK-like stencil (simplified)
    phi = np.pad(field, 2, mode='edge')
    for j in range(ny):
        for i in range(nx):
            if u[j, i] >= 0:
                phi_e = (3/8)*phi[j+2, i+3] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+2, i+1]
                phi_w = (3/8)*phi[j+2, i+2] + (6/8)*phi[j+2, i+1] - (1/8)*phi[j+2, i]
            else:
                phi_e = (3/8)*phi[j+2, i+1] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+2, i+3]
                phi_w = (3/8)*phi[j+2, i] + (6/8)*phi[j+2, i+1] - (1/8)*phi[j+2, i+2]
            if v[j, i] >= 0:
                phi_n = (3/8)*phi[j+3, i+2] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+1, i+2]
                phi_s = (3/8)*phi[j+2, i+2] + (6/8)*phi[j+1, i+2] - (1/8)*phi[j, i+2]
            else:
                phi_n = (3/8)*phi[j+1, i+2] + (6/8)*phi[j+2, i+2] - (1/8)*phi[j+3, i+2]
                phi_s = (3/8)*phi[j, i+2] + (6/8)*phi[j+1, i+2] - (1/8)*phi[j+2, i+2]
            flux_x_e = u[j, i] * phi_e
            flux_x_w = u[j, i] * phi_w
            flux_y_n = v[j, i] * phi_n
            flux_y_s = v[j, i] * phi_s
            conv[j, i] = (flux_x_e - flux_x_w)/dx + (flux_y_n - flux_y_s)/dy
    return conv

__all__ = ['advect_upwind', 'advect_quick']
