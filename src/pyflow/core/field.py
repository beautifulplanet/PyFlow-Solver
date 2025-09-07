from typing import Any

import numpy as np


class Field:
    """Container for a scalar field with simple boundary metadata."""

    def __init__(self, data: Any, name: str, location: str = 'cell_center', bc: dict[str, tuple[str, float]] | None = None) -> None:
        self.data: np.ndarray = np.array(data)
        self.name: str = name
        self.location: str = location
        self.bc: dict[str, tuple[str, float]] = bc or {}

    def set_boundary(self, bc_type: str, value: float, face: str) -> None:
        self.bc[face] = (bc_type, value)

    def get_boundary(self, face: str) -> tuple[str, float] | None:
        return self.bc.get(face, None)

    def copy(self) -> 'Field':
        return Field(self.data.copy(), self.name, self.location, self.bc.copy())
