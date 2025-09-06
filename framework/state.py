from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class Mesh:
    def __init__(self, nx: int = 17, ny: int = 17, lx: float = 1.0, ly: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self._dx = lx / max(nx - 1, 1)
        self._dy = ly / max(ny - 1, 1)

    def dx(self) -> float:
        return self._dx

    def dy(self) -> float:
        return self._dy


class SolverState:
    def __init__(self, mesh: Mesh | None = None, fields: Dict[str, np.ndarray] | None = None, nu: float = 0.01, rho: float = 1.0):
        self.mesh = mesh if mesh is not None else Mesh()
        self.fields: Dict[str, np.ndarray] = fields if fields is not None else {}
        self.nu = nu
        self.rho = rho
        self.time = 0.0

    def shape(self) -> Tuple[int, int]:
        return (self.mesh.nx, self.mesh.ny)

    def require_field(self, name: str, shape: Tuple[int, int] | None, init: float = 0.0) -> np.ndarray:
        if name in self.fields:
            arr = self.fields[name]
            if shape is not None and arr.shape != shape:
                raise ValueError(f"Field '{name}' existing shape {arr.shape} != requested {shape}")
            return arr
        if shape is None:
            raise ValueError("shape must be provided for new field")
        arr = np.full(shape, init, dtype=float)
        self.fields[name] = arr
        return arr

    def advance_time(self, dt: float) -> None:
        self.time += float(dt)

    # Convenience properties
    @property
    def u(self) -> np.ndarray:
        return self.require_field("u", self.shape())

    @property
    def v(self) -> np.ndarray:
        return self.require_field("v", self.shape())

    @property
    def p(self) -> np.ndarray:
        return self.require_field("p", self.shape())


__all__ = ["Mesh", "SolverState"]

