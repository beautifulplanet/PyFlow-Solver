from __future__ import annotations
import numpy as np
from ..core.ghost_fields import interior_view, State

def divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny, nx = u.shape
    div = np.zeros_like(u)
    div[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,0:-2])/(2*dx) + (v[2:,1:-1]-v[0:-2,1:-1])/(2*dy)
    if nx>1:
        div[:,0] = (u[:,1]-u[:,0])/dx
        div[:,-1] = (u[:,-1]-u[:,-2])/dx
    if ny>1:
        div[0,:] += (v[1,:]-v[0,:])/dy
        div[-1,:] += (v[-1,:]-v[-2,:])/dy
    return div

def gradient(p: np.ndarray, dx: float, dy: float):
    ny, nx = p.shape
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    if nx>2:
        dpdx[:,1:-1] = (p[:,2:] - p[:,0:-2])/(2*dx)
    if ny>2:
        dpdy[1:-1,:] = (p[2:,:] - p[0:-2,:])/(2*dy)
    if nx>1:
        dpdx[:,0] = (p[:,1]-p[:,0])/dx
        dpdx[:,-1] = (p[:,-1]-p[:,-2])/dx
    if ny>1:
        dpdy[0,:] = (p[1,:]-p[0,:])/dy
        dpdy[-1,:] = (p[-1,:]-p[-2,:])/dy
    return dpdx, dpdy

def apply_pressure_correction(state: State, p: np.ndarray, dt: float, dx: float, dy: float) -> None:
    interior_view(state.fields['p'])[:] = p
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    dpdx, dpdy = gradient(p, dx, dy)
    ui[:] -= dt * dpdx
    vi[:] -= dt * dpdy

_orig_divergence = divergence

def divergence_dispatch(arg0, *rest, **kwargs):  # pragma: no cover
    if hasattr(arg0, 'fields'):
        dx = kwargs.get('dx', 1.0); dy = kwargs.get('dy', 1.0)
        ui = interior_view(arg0.fields['u']); vi = interior_view(arg0.fields['v'])
        return _orig_divergence(ui, vi, dx, dy)
    return _orig_divergence(arg0, *rest, **kwargs)

# Provide backwards-compatible name
divergence = divergence_dispatch  # type: ignore[assignment]

__all__ = ['divergence', 'gradient', 'apply_pressure_correction']
