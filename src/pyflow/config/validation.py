from __future__ import annotations

"""Configuration validation utilities.

Avoids external dependency bloat; provides a validate_config function that
raises ValueError on invalid user-supplied parameters.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ValidatedConfig:
    nx: int
    ny: int
    Re: float
    lid_velocity: float
    cfl_target: float
    cfl_growth: float
    lin_tol: float
    lin_maxiter: int

def validate_config(cfg: Any) -> None:
    def ensure(cond: bool, msg: str):
        if not cond:
            raise ValueError(f"Config validation failed: {msg}")
    nx = getattr(cfg, 'nx', None); ny = getattr(cfg, 'ny', None)
    ensure(isinstance(nx, int) and nx > 1, 'nx must be int > 1')
    ensure(isinstance(ny, int) and ny > 1, 'ny must be int > 1')
    Re = getattr(cfg, 'Re', 100.0)
    ensure(Re > 0, 'Re must be positive')
    cfl = getattr(cfg, 'cfl_target', 0.5)
    ensure(0 < cfl <= 1.0, 'cfl_target in (0,1]')
    cflg = getattr(cfg, 'cfl_growth', 1.0)
    ensure(cflg >= 1.0 and cflg <= 1.2, 'cfl_growth in [1.0,1.2] (safety bound)')
    lin_tol = getattr(cfg, 'lin_tol', 1e-10)
    ensure(lin_tol > 0, 'lin_tol must be > 0')
    lin_max = getattr(cfg, 'lin_maxiter', 1)
    ensure(lin_max > 0, 'lin_maxiter must be > 0')

__all__ = ["validate_config"]