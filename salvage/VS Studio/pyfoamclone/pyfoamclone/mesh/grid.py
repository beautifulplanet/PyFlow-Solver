import numpy as np
from typing import Tuple


class Grid:
    """Generates a 2D structured Cartesian mesh (uniform)."""

    def __init__(self, nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> None:
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        self.nx: int = nx
        self.ny: int = ny
        self.lx: float = lx
        self.ly: float = ly
        self.dx: float = lx / nx
        self.dy: float = ly / ny
        self.xc = np.linspace(self.dx / 2, lx - self.dx / 2, nx)
        self.yc = np.linspace(self.dy / 2, ly - self.dy / 2, ny)
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.volumes = np.full((nx, ny), self.dx * self.dy)
        self.xf = np.linspace(0, lx, nx + 1)
        self.yf = np.linspace(0, ly, ny + 1)
        self.Xf, self.Yf = np.meshgrid(self.xf, self.yf, indexing="ij")
        self.area_x = np.full((nx + 1, ny), self.dy)
        self.area_y = np.full((nx, ny + 1), self.dx)

    def shape(self) -> Tuple[int, int]:
        return self.nx, self.ny
