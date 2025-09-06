"""Minimal framework package exposing canonical Mesh/SolverState and solver hooks.

The goal is to provide a single set of concrete classes so tests do not see
stub vs real type mismatches. Implementations are intentionally lightweight.
"""

from .state import Mesh, SolverState  # noqa: F401
from .projection_solver import (
    projection_step,
    pressure_rhs_unscaled,
    solve_pressure_poisson_unscaled,
)
from .advection import cfl_dt  # noqa: F401
from .steady_diffusion import solve_steady_diffusion  # noqa: F401

__all__ = [
    "Mesh",
    "SolverState",
    "projection_step",
    "pressure_rhs_unscaled",
    "solve_pressure_poisson_unscaled",
    "cfl_dt",
    "solve_steady_diffusion",
]
