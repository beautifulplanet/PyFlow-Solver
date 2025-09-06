"""
tests/test_structured_grid.py
Unit test for StructuredGrid class.
"""
import unittest
import numpy as np
from grid.structured import StructuredGrid

class TestStructuredGrid(unittest.TestCase):
    def test_grid_properties(self):
        Lx, Ly = 2.0, 1.0
        Nx, Ny = 5, 3
        grid = StructuredGrid(Lx, Ly, Nx, Ny)
        self.assertAlmostEqual(grid.dx, 0.5)
        self.assertAlmostEqual(grid.dy, 0.5)
        self.assertEqual(grid.x.shape, (Nx,))
        self.assertEqual(grid.y.shape, (Ny,))
        self.assertEqual(grid.X.shape, (Ny, Nx))
        self.assertEqual(grid.Y.shape, (Ny, Nx))

    def test_cell_centers(self):
        Lx, Ly = 2.0, 1.0
        Nx, Ny = 5, 3
        grid = StructuredGrid(Lx, Ly, Nx, Ny)
        X_cc, Y_cc = grid.cell_centers()
        self.assertEqual(X_cc.shape, (Ny-2, Nx-2))
        self.assertEqual(Y_cc.shape, (Ny-2, Nx-2))
        # Check that cell centers are within domain (not on boundaries)
        self.assertTrue(np.all(X_cc > 0))
        self.assertTrue(np.all(X_cc < Lx))
        self.assertTrue(np.all(Y_cc > 0))
        self.assertTrue(np.all(Y_cc < Ly))

    def test_spacing(self):
        grid = StructuredGrid(1.0, 1.0, 11, 11)
        dx, dy = grid.spacing()
        self.assertAlmostEqual(dx, 0.1)
        self.assertAlmostEqual(dy, 0.1)

if __name__ == "__main__":
    unittest.main()
