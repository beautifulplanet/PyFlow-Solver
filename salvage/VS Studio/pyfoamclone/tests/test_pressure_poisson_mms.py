import numpy as np
import pytest
from pyfoamclone.mesh.grid import make_grid
from pyfoamclone.solvers.pressure_solver import assemble_negative_laplacian, solve_pressure_poisson

def analytic_pressure(x, y):
    # p(x, y) = cos(2pi x) * cos(2pi y)
    return np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

def analytic_rhs(x, y):
    # -Laplacian(p) = 8 * pi^2 * cos(2pi x) * cos(2pi y)
    return 8 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

def test_pressure_poisson_manufactured_solution():
    # Test for grid refinement
    errors = []
    for n in [16, 32, 64]:
        nx = ny = n
        dx = dy = 1.0 / (n - 1)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        p_exact = analytic_pressure(X, Y)
        rhs = analytic_rhs(X, Y)
        # Flatten for solver
        rhs_flat = rhs.ravel()
        # Assemble Laplacian
        A = assemble_negative_laplacian(nx, ny, dx, dy)
        # Solve
        p_num_flat = solve_pressure_poisson(A, rhs_flat, nx, ny)
        p_num = p_num_flat.reshape((nx, ny))
        # Remove mean (Neumann BCs)
        p_num -= np.mean(p_num)
        p_exact -= np.mean(p_exact)
        # Compute L2 error
        error = np.linalg.norm(p_num - p_exact) / np.sqrt(nx * ny)
        errors.append(error)
        print(f"n={n}: L2 error={error}")
    # Check convergence: error should decrease ~4x for each 2x refinement (second order)
    assert errors[1] < errors[0] * 0.3
    assert errors[2] < errors[1] * 0.3
    assert errors[-1] < 1e-3, f"L2 error too high: {errors[-1]}"
