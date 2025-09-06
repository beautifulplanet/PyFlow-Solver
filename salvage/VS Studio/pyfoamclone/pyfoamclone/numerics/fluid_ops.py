from __future__ import annotations
import numpy as np
from ..core.ghost_fields import interior_view, State

"""Core numerical operators with explicit spacing.

New explicit signatures:
    divergence(u, v, dx, dy)
    gradient(p, dx, dy) -> (dpdx, dpdy)
    laplacian(p, dx, dy)

All array arguments are INTERIOR (ny, nx) arrays (no ghost cells). Boundary rows/cols
use one-sided first-order differences (implemented by zero-padding / copying neighboring values)
so that returned shapes remain (ny, nx).

Existing solver code has been updated to extract interior arrays explicitly. Legacy wrappers
are provided for transition but tests now target the new signatures.
"""

def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny, nx = u.shape
    div = np.zeros_like(u)
    # central differences interior
    div[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,0:-2]) / (2.0*dx) + (v[2:,1:-1] - v[0:-2,1:-1]) / (2.0*dy)
    # simple one-sided at boundaries (could be refined later)
    if nx > 1:
        div[:,0] = (u[:,1] - u[:,0]) / dx
        div[:,-1] = (u[:,-1] - u[:,-2]) / dx
    if ny > 1:
        div[0,:] += (v[1,:] - v[0,:]) / dy
        div[-1,:] += (v[-1,:] - v[-2,:]) / dy
    return div


def gradient(p: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = p.shape
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    if nx > 2:
        dpdx[:,1:-1] = (p[:,2:] - p[:,0:-2]) / (2.0*dx)
    if ny > 2:
        dpdy[1:-1,:] = (p[2:,:] - p[0:-2,:]) / (2.0*dy)
    # one-sided boundaries
    if nx > 1:
        dpdx[:,0] = (p[:,1] - p[:,0]) / dx
        dpdx[:,-1] = (p[:,-1] - p[:,-2]) / dx
    if ny > 1:
        dpdy[0,:] = (p[1,:] - p[0,:]) / dy
        dpdy[-1,:] = (p[-1,:] - p[-2,:]) / dy
    return dpdx, dpdy


def laplacian(p: np.ndarray, dx: float, dy: float) -> np.ndarray:
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


# Legacy wrappers (will be removed in later cleanup) ---------------------------------
def legacy_divergence_state(state: State, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:  # pragma: no cover
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    return divergence(ui, vi, dx, dy)

def legacy_grad_p_state(state: State, dx: float = 1.0, dy: float = 1.0):  # pragma: no cover
    pi = interior_view(state.fields['p'])
    return gradient(pi, dx, dy)

def apply_pressure_correction(state: State, p: np.ndarray, dt: float, dx: float, dy: float) -> None:
    pi = p
    interior_view(state.fields['p'])[:] = pi
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    dpdx, dpdy = gradient(pi, dx, dy)
    ui[:] = ui - dt * dpdx
    vi[:] = vi - dt * dpdy

# Public API compatibility: allow divergence(state, ...) old style
_divergence_arrays = divergence
def divergence_dispatch(arg0, *rest, **kwargs):  # pragma: no cover - wrapper glue
    if hasattr(arg0, 'fields'):  # State
        dx = kwargs.get('dx', 1.0)
        dy = kwargs.get('dy', 1.0)
        return legacy_divergence_state(arg0, dx=dx, dy=dy)
    else:
        return _divergence_arrays(arg0, *rest, **kwargs)

# Expose wrapper under original expected name for existing tests
divergence = divergence_dispatch
