from __future__ import annotations
import numpy as np
from ..core.ghost_fields import interior_view, State

def _divergence_impl(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the discrete divergence of a 2D vector field (u, v) using central differences in the interior
    and one-sided differences at boundaries.
    """
    ny, nx = u.shape
    div = np.zeros_like(u)
    # Interior: central differences
    div[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,0:-2]) / (2.0*dx) + (v[2:,1:-1] - v[0:-2,1:-1]) / (2.0*dy)
    # Left boundary (excluding corners)
    div[1:-1,0] = (u[1:-1,1] - u[1:-1,0]) / dx + (v[2:,0] - v[0:-2,0]) / (2.0*dy)
    # Top boundary (excluding corners)
    div[0,1:-1] = (u[0,2:] - u[0,:-2]) / (2.0*dx) + (v[1,1:-1] - v[0,1:-1]) / dy
    # Bottom boundary (excluding corners)
    div[-1,1:-1] = (u[-1,2:] - u[-1,:-2]) / (2.0*dx) + (v[-1,1:-1] - v[-2,1:-1]) / dy
    # Corners
    div[0,0] = (u[0,1] - u[0,0]) / dx + (v[1,0] - v[0,0]) / dy
    div[0,-1] = (u[0,-1] - u[0,-2]) / dx + (v[1,-1] - v[0,-1]) / dy
    div[-1,0] = (u[-1,1] - u[-1,0]) / dx + (v[-1,0] - v[-2,0]) / dy
    div[-1,-1] = (u[-1,-1] - u[-1,-2]) / dx + (v[-1,-1] - v[-2,-1]) / dy
    return div

def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Public interface for computing the discrete divergence of a 2D vector field.
    Args:
        u: x-velocity array
        v: y-velocity array
        dx: grid spacing in x
        dy: grid spacing in y
    Returns:
        2D array of divergence values
    """
    return _divergence_impl(u, v, dx, dy)

def gradient(p: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient of a scalar field p using central differences in the interior and one-sided at boundaries.
    Args:
        p: 2D scalar field
        dx: grid spacing in x
        dy: grid spacing in y
    Returns:
        Tuple of (dpdx, dpdy) arrays
    """
    ny, nx = p.shape
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    # Interior: central differences
    if nx > 2:
        dpdx[:,1:-1] = (p[:,2:] - p[:,0:-2]) / (2.0*dx)
    if ny > 2:
        dpdy[1:-1,:] = (p[2:,:] - p[0:-2,:]) / (2.0*dy)
    # Left/right boundaries (one-sided)
    dpdx[:,0] = (p[:,1] - p[:,0]) / dx
    dpdx[:,-1] = (p[:,-1] - p[:,-2]) / dx
    # Top/bottom boundaries (one-sided)
    dpdy[0,:] = (p[1,:] - p[0,:]) / dy
    dpdy[-1,:] = (p[-1,:] - p[-2,:]) / dy
    return dpdx, dpdy

def laplacian(p: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the Laplacian of a scalar field p using central differences in the interior and crude one-sided at boundaries.
    Args:
        p: 2D scalar field
        dx: grid spacing in x
        dy: grid spacing in y
    Returns:
        2D array of Laplacian values
    """
    ny, nx = p.shape
    lap = np.zeros_like(p)
    if nx > 2 and ny > 2:
        lap[1:-1,1:-1] = (
            (p[1:-1,2:] - 2.0*p[1:-1,1:-1] + p[1:-1,0:-2]) / (dx*dx) +
            (p[2:,1:-1] - 2.0*p[1:-1,1:-1] + p[0:-2,1:-1]) / (dy*dy)
        )
    # crude one-sided boundaries (copy nearest interior second derivative estimate)
    lap[0,:] = lap[1,:]
    lap[-1,:] = lap[-2,:]
    lap[:,0] = lap[:,1]
    lap[:,-1] = lap[:,-2]
    return lap

def legacy_divergence_state(state: State, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Compute divergence from a State object using interior views of u and v fields.
    """
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    return _divergence_impl(ui, vi, dx, dy)

def legacy_grad_p_state(state: State, dx: float = 1.0, dy: float = 1.0):
    """
    Compute gradient of pressure from a State object using interior view of p field.
    """
    pi = interior_view(state.fields['p'])
    return gradient(pi, dx, dy)

def apply_pressure_correction(state: State, p: np.ndarray, dt: float, dx: float, dy: float) -> None:
    """
    Apply pressure correction to the velocity fields in the State object.
    Args:
        state: State object with fields 'u', 'v', 'p'
        p: pressure field to use for correction
        dt: timestep
        dx: grid spacing in x
        dy: grid spacing in y
    """
    pi = p
    interior_view(state.fields['p'])[:] = pi
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    dpdx, dpdy = gradient(pi, dx, dy)
    ui[:] = ui - dt * dpdx
    vi[:] = vi - dt * dpdy

