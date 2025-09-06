"""Core solver state dataclass.

Responsibility:
- Hold mesh, field arrays, material properties, time-stepping info, metadata.
- Provide sanity checks (dimensions, dtype) and controlled mutation points.

NOTE: Keep minimal now; extend cautiously (add turbulence, etc. later via optional mixins).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

Array = np.ndarray

@dataclass
class Mesh:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def dx(self) -> float:
        return self.lx / (self.nx - 1)

    def dy(self) -> float:
        return self.ly / (self.ny - 1)

@dataclass
class SolverState:
    mesh: Mesh
    fields: Dict[str, Array]
    time: float = 0.0
    step: int = 0
    rho: float = 1.0
    nu: float = 0.01  # kinematic viscosity
    metadata: Dict[str, Any] = field(default_factory=dict)

    def require_field(self, name: str, shape: tuple[int, int]):
        if name not in self.fields:
            self.fields[name] = np.zeros(shape, dtype=float)
        arr = self.fields[name]
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' shape mismatch: have {arr.shape} expected {shape}")
        return arr

    def shape(self) -> tuple[int, int]:
        return (self.mesh.nx, self.mesh.ny)

    def advance_time(self, dt: float):
        self.time += dt
        self.step += 1
        self.metadata['last_dt'] = dt

    def clone_shallow(self) -> 'SolverState':
        # Shallow copy fields (views) for what-if analyses; deep copy only if needed.
        return SolverState(mesh=self.mesh, fields=dict(self.fields), time=self.time, step=self.step, rho=self.rho, nu=self.nu, metadata=dict(self.metadata))
