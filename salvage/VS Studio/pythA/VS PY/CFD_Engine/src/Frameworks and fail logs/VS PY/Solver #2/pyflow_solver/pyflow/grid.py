import numpy as np

class Grid:
    """
    Manages the creation and properties of a 2D computational grid.
    This class creates a uniform, staggered grid.
    """
    def __init__(self, NPOINTS: int, L: float):
        self.NPOINTS = NPOINTS
        self.L = L
        self.dx = L / (NPOINTS - 1)
        self.dy = L / (NPOINTS - 1)
        x = np.linspace(0, L, NPOINTS, dtype=np.float64)
        y = np.linspace(0, L, NPOINTS, dtype=np.float64)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Additional properties for compatibility with visualization
        self.Nx = self.Ny = NPOINTS
        self.Lx = self.Ly = L

    def __repr__(self):
        return (f"Grid(NPOINTS={self.NPOINTS}, L={self.L}, dx={self.dx}, dy={self.dy})")
