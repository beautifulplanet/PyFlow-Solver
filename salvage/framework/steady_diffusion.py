"""Steady diffusion solvers (Jacobi / SOR) for -Lap(u) = f/nu with Dirichlet boundaries.

Used for convergence and performance regression tests.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def solve_steady_diffusion(u: np.ndarray, f: np.ndarray, dx: float, dy: float, nu: float,
                           tol: float = 1e-8, max_iter: int = 100_000, method: str = 'jacobi', omega: float = 1.8,
                           progress: bool = False, progress_every: int = 2000) -> Tuple[int, float]:
    nx, ny = u.shape
    hx2, hy2 = dx*dx, dy*dy
    denom = 2*(hx2 + hy2)
    inv_nu = 1.0 / nu

    if method not in ('jacobi','sor'):
        raise ValueError("method must be 'jacobi' or 'sor'")

    if method == 'jacobi':
        work = u.copy()
    residual = 1e30
    for it in range(max_iter):
        if method == 'jacobi':
            # Discrete update for -Lap(u) = f/nu:
            # u_ij = ( hy2*(u_{i+1,j}+u_{i-1,j}) + hx2*(u_{i,j+1}+u_{i,j-1}) + f_ij*hx2*hy2/nu ) / (2*(hx2+hy2))
            work[1:-1,1:-1] = (
                (u[2:,1:-1] + u[:-2,1:-1]) * hy2 +
                (u[1:-1,2:] + u[1:-1,:-2]) * hx2 +
                f[1:-1,1:-1] * hx2 * hy2 * inv_nu
            ) / denom
            u, work = work, u
        else:  # SOR (lexicographic sweep)
            # Clamp omega to safe range to avoid overflow
            omega = min(max(omega, 1.0), 1.95)
            u[1:-1,1:-1] = (1-omega)*u[1:-1,1:-1] + omega * (
                (u[2:,1:-1] + u[:-2,1:-1]) * hy2 +
                (u[1:-1,2:] + u[1:-1,:-2]) * hx2 +
                f[1:-1,1:-1] * hx2 * hy2 * inv_nu
            ) / denom
            # Check for overflow
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                import warnings
                warnings.warn("SOR produced NaN or Inf; omega may be too large.")
                return it, float('inf')
        # residual every progress_every or first few iters for speed
        if it % progress_every == 0 or it < 5:
            lap = ((u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/hx2 + (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/hy2)
            r = -lap - f[1:-1,1:-1]*inv_nu
            new_resid = float(np.max(np.abs(r)))
            # simple divergence safeguard: if residual exploded, abort early
            if new_resid > 1e12:
                return it, new_resid
            residual = new_resid
            if progress and (it % progress_every == 0):
                print(f"iter={it} resid={residual:.3e}")
            if residual < tol:
                return it, residual
    return max_iter, residual
