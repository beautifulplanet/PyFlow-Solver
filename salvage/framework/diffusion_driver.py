"""Simple explicit diffusion time-stepper (for validation & residual testing).

u_t = nu (u_xx + u_yy) + S
Dirichlet 0 boundary for now.
NOT production-stable (no CFL adaptive dt yet) – used to validate residual plumbing.
"""
from __future__ import annotations
import numpy as np
from .state import Mesh, SolverState
from .residuals import diffusion_residual, laplacian


def step_diffusion(state: SolverState, source: np.ndarray, dt: float):
    nx, ny = state.shape()
    u = state.require_field('u', (nx, ny))
    dx, dy = state.mesh.dx(), state.mesh.dy()
    # Explicit Euler interior update
    lap = np.zeros((nx-2, ny-2))
    lap[:, :] = laplacian(u, dx, dy)
    u[1:-1, 1:-1] += dt * (state.nu * lap + source[1:-1, 1:-1])
    state.advance_time(dt)
    return diffusion_residual(u, source, dx, dy)


def init_state(nx: int = 33, ny: int = 33, nu: float = 0.01) -> SolverState:
    mesh = Mesh(nx=nx, ny=ny)
    fields = {'u': np.zeros((nx, ny), dtype=float)}
    return SolverState(mesh=mesh, fields=fields, nu=nu)
