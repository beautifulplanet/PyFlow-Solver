"""Grid abstraction (initial spike).

Provides a minimal UniformGrid class today while defining an interface surface
that can later be satisfied by stretched or unstructured meshes. The solver
currently assumes uniform spacing; this layer centralizes that assumption.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class UniformGrid:
    nx: int
    ny: int
    lx: float
    ly: float

    @property
    def dx(self) -> float:
        return self.lx / max(self.nx - 1, 1)

    @property
    def dy(self) -> float:
        return self.ly / max(self.ny - 1, 1)

    # Future extension points:
    # def cell_volume(self, i, j): ...
    # def face_area(self, i, j, dir): ...

__all__ = ["UniformGrid"]