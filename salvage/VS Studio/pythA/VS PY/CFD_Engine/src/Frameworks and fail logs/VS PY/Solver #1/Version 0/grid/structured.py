"""
grid/structured.py
Structured grid generation for CFD solver.
"""

import numpy as np

class StructuredGrid:
    """
    Represents a 2D structured grid for CFD simulations.
    """
    def __init__(self, Lx, Ly, Nx, Ny):
        """
        Initialize the grid.
        Args:
            Lx (float): Length of the domain in x-direction.
            Ly (float): Length of the domain in y-direction.
            Nx (int): Number of grid points in x-direction.
            Ny (int): Number of grid points in y-direction.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def cell_centers(self):
        """
        Returns the coordinates of cell centers (excluding boundaries).
        """
        x_cc = self.x[1:-1]
        y_cc = self.y[1:-1]
        X_cc, Y_cc = np.meshgrid(x_cc, y_cc)
        return X_cc, Y_cc

    def spacing(self):
        """
        Returns grid spacing (dx, dy).
        """
        return self.dx, self.dy
