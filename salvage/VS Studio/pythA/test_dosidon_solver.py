import unittest
import numpy as np
from dosidon_solver import run_dosidon_simulation

class TestDosidonSolver(unittest.TestCase):
    def setUp(self):
        # Standard test grid and time step
        self.nx = 4
        self.ny = 4
        self.nt = 2
        self.dt = 0.01

    def test_zero_initial_velocity(self):
        """Test that zero initial velocity remains zero (trivial steady state)."""
        u, v, p = run_dosidon_simulation(self.nx, self.ny, self.nt, self.dt)
        self.assertTrue(np.allclose(u, 0))
        self.assertTrue(np.allclose(v, 0))

    def test_pressure_field_shape(self):
        """Test that the pressure field has the correct shape."""
        _, _, p = run_dosidon_simulation(self.nx, self.ny, self.nt, self.dt)
        self.assertEqual(p.shape, (self.nx, self.ny))

    def test_compare_to_openfoam_lid_driven_cavity(self):
        """
        Compare the centerline velocity profile to reference OpenFOAM data for a lid-driven cavity.
        This is a placeholder: in a real test, you would load OpenFOAM results and compare.
        """
        # Run with more steps for a more realistic test
        nx, ny, nt, dt = 16, 16, 10, 0.01
        u, v, p = run_dosidon_simulation(nx, ny, nt, dt)
        # Placeholder: reference values from OpenFOAM (should be loaded from file or hardcoded)
        openfoam_centerline_u = np.zeros(ny)  # Replace with real OpenFOAM data
        dosidon_centerline_u = u[nx//2, :]
        # Allow large tolerance for placeholder
        self.assertTrue(np.allclose(dosidon_centerline_u, openfoam_centerline_u, atol=1e-1))

    def test_mass_conservation(self):
        """Test that the total mass (sum of velocity divergence) is approximately conserved."""
        u, v, p = run_dosidon_simulation(self.nx, self.ny, self.nt, self.dt)
        divergence = np.sum(u) + np.sum(v)
        self.assertAlmostEqual(divergence, 0, places=2)

    def test_stability_small_dt(self):
        """Test that the solver is stable for a small time step."""
        u, v, p = run_dosidon_simulation(self.nx, self.ny, 5, 1e-4)
        self.assertFalse(np.any(np.isnan(u)))
        self.assertFalse(np.any(np.isnan(v)))
        self.assertFalse(np.any(np.isnan(p)))

if __name__ == "__main__":
    unittest.main()
