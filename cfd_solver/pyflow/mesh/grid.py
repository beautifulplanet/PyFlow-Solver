import numpy as np

class Grid:
    """
    Generates a 2D structured Cartesian mesh.
    Computes cell centers, face centers, cell volumes, and face area vectors.
    """
    def __init__(self, nx, ny, lx=1.0, ly=1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.xc = np.linspace(self.dx/2, lx - self.dx/2, nx)
        self.yc = np.linspace(self.dy/2, ly - self.dy/2, ny)
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='ij')
        self.volumes = np.full((nx, ny), self.dx * self.dy)
        # Face centers (for staggered grid or fluxes)
        self.xf = np.linspace(0, lx, nx+1)
        self.yf = np.linspace(0, ly, ny+1)
        self.Xf, self.Yf = np.meshgrid(self.xf, self.yf, indexing='ij')
        # Face area vectors (for 2D, just dx or dy)
        self.area_x = np.full((nx+1, ny), self.dy)
        self.area_y = np.full((nx, ny+1), self.dx)
