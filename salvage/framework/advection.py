"""Advection kernels and CFL time step controller.

Implements first-order upwind convective derivatives for u and v components
and a combined advective + diffusion stability dt estimate.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def upwind_derivative_x(phi: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    # phi shape (nx, ny); u velocities at cell centers same shape.
    # One-sided based on sign of u at i,j using backward/forward difference.
    d = np.zeros_like(phi)
    # interior only
    pos = u[1:-1,1:-1] >= 0
    # backward difference for positive
    d[1:-1,1:-1][pos] = (phi[1:-1,1:-1][pos] - phi[0:-2,1:-1][pos]) / dx
    # forward difference for negative
    neg = ~pos
    d[1:-1,1:-1][neg] = (phi[2:,1:-1][neg] - phi[1:-1,1:-1][neg]) / dx
    return d

def upwind_derivative_y(phi: np.ndarray, v: np.ndarray, dy: float) -> np.ndarray:
    d = np.zeros_like(phi)
    pos = v[1:-1,1:-1] >= 0
    d[1:-1,1:-1][pos] = (phi[1:-1,1:-1][pos] - phi[1:-1,0:-2][pos]) / dy
    neg = ~pos
    d[1:-1,1:-1][neg] = (phi[1:-1,2:][neg] - phi[1:-1,1:-1][neg]) / dy
    return d

def convective_terms(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Cu, Cv) where Cu = u*du/dx + v*du/dy, similarly for v.
    Uses first-order upwind derivatives. Zero on boundaries (Dirichlet assumed)."""
    du_dx = upwind_derivative_x(u, u, dx)
    du_dy = upwind_derivative_y(u, v, dy)
    dv_dx = upwind_derivative_x(v, u, dx)
    dv_dy = upwind_derivative_y(v, v, dy)
    Cu = u * du_dx + v * du_dy
    Cv = u * dv_dx + v * dv_dy
    return Cu, Cv

def cfl_dt(u: np.ndarray, v: np.ndarray, dx: float, dy: float, nu: float, cfl: float = 0.5, safety: float = 0.9) -> float:
    """Compute stable dt from advective & diffusion limits.
    advective: dt_a <= cfl * min(dx/max|u|, dy/max|v|) (ignore component with zero max)
    diffusion (explicit): dt_d <= 0.25 * min(dx^2, dy^2)/nu (2D stability heuristic).
    Returns safety * min(dt_a, dt_d) (skips advective if velocities near zero).
    """
    umax = float(np.max(np.abs(u)))
    vmax = float(np.max(np.abs(v)))
    adv_limits = []
    if umax > 1e-14:
        adv_limits.append(dx/umax)
    if vmax > 1e-14:
        adv_limits.append(dy/vmax)
    if adv_limits:
        dt_adv = cfl * min(adv_limits)
    else:
        dt_adv = float('inf')
    dt_diff = 0.25 * min(dx*dx, dy*dy) / max(nu, 1e-14)
    dt = min(dt_adv, dt_diff)
    if dt == float('inf'):
        dt = dt_diff
    return safety * dt
