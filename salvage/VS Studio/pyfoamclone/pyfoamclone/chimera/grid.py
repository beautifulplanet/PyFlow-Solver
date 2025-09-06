import numpy as np

class Grid:
    def __init__(self, nx, ny, lx, ly):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.x = np.linspace(0, lx, nx+1)
        self.y = np.linspace(0, ly, ny+1)
        # For test compatibility:
        self.xp = np.linspace(0, lx, nx)
        self.yp = np.linspace(0, ly, ny)
        self.xu = np.linspace(0, lx, nx+1)
        self.yv = np.linspace(0, ly, ny+1)
