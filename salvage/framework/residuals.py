"""Physical residual computations (initial scaffolding).

Residuals should reflect equation imbalance, not simple field deltas.
Future: momentum, continuity, energy residuals with consistent nondimensionalization.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

Array = np.ndarray


def laplacian(u: Array, dx: float, dy: float) -> Array:
    return (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )


def diffusion_residual(u: Array, source: Array, dx: float, dy: float) -> Dict[str, float]:
    r = -laplacian(u, dx, dy) - source[1:-1, 1:-1]
    linf = float(np.max(np.abs(r)))
    l2 = float(np.sqrt(np.mean(r**2)))
    return {"Linf": linf, "L2": l2}


def continuity_residual(u: Array, v: Array, dx: float, dy: float) -> Dict[str, float]:
    """Compute discrete divergence residual (incompressibility) on interior."""
    div = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
    linf = float(np.max(np.abs(div))) if div.size else 0.0
    l2 = float(np.sqrt(np.mean(div**2))) if div.size else 0.0
    return {"Linf": linf, "L2": l2}

def momentum_residual(u: Array, v: Array, p: Array, nu: float, dx: float, dy: float) -> Dict[str, float]:
    """Placeholder momentum equation residual (diffusion + pressure gradient only)."""
    # Diffusion terms
    lap_u = laplacian(u, dx, dy)
    lap_v = laplacian(v, dx, dy)
    # Pressure gradients (central)
    dpdx = (p[2:,1:-1]-p[:-2,1:-1])/(2*dx)
    dpdy = (p[1:-1,2:]-p[1:-1,:-2])/(2*dy)
    ru = -dpdx + nu*lap_u
    rv = -dpdy + nu*lap_v
    linf = float(max(np.max(np.abs(ru)), np.max(np.abs(rv)))) if ru.size else 0.0
    l2 = float(np.sqrt(0.5*(np.mean(ru**2)+np.mean(rv**2)))) if ru.size else 0.0
    return {"Linf": linf, "L2": l2}
