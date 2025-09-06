import numpy as np
from pyflow.mesh.grid import Grid

def test_grid_basic():
    grid = Grid(2, 2, lx=1.0, ly=1.0)
    assert grid.nx == 2 and grid.ny == 2
    assert np.isclose(grid.dx, 0.5)
    assert np.isclose(grid.dy, 0.5)
    assert grid.Xc.shape == (2, 2)
    assert grid.volumes.shape == (2, 2)
    assert np.allclose(grid.volumes, 0.25)
    assert grid.Xf.shape == (3, 3)
    assert grid.area_x.shape == (3, 2)
    assert grid.area_y.shape == (2, 3)
