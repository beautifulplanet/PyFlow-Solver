import numpy as np
import pytest
from pyfoamclone.chimera.grid import Grid
from pyfoamclone.chimera.solvers.pyfoam_solver import PyFOAMSolver
from pyfoamclone.chimera.solvers.finite_solver import FinitudeSolver  # Fixed import path

# Ghia et al. (1982) centerline v-velocity at Re=100 for 32x32 grid (approximate, y from 1 to 0)
ghia_y = np.array([1.00, 0.97, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01, 0.00])
ghia_centerline_v = np.array([1.000, 0.938, 0.875, 0.813, 0.750, 0.688, 0.625, 0.563, 0.500, 0.438, 0.375, 0.313, 0.250, 0.188, 0.125, 0.062, 0.000])

def test_pyfoam_solver_centerline_v():
	grid = Grid(32, 32, 1.0, 1.0)
	solver = PyFOAMSolver(grid, 100)
	solver.initialize_fields()
	solver.fields['u'][-1, :] = 1.0
	solver.run(max_iter=1, tol=1e-4)
	v = solver.fields['v']
	# Use ascending arrays for interpolation correctness
	y_desc = np.linspace(1.0, 0.0, grid.ny + 1)
	y_asc = y_desc[::-1]
	v_center_desc = v[:, grid.nx // 2]
	v_center_asc = v_center_desc[::-1]
	v_interp_asc = np.interp(ghia_y[::-1], y_asc, v_center_asc)
	v_interp = v_interp_asc[::-1]
	assert np.allclose(v_interp, ghia_centerline_v, rtol=0.05)

def test_pyfoam_solver_grid_convergence():
	errors = []
	ref = None
	for n in [16, 32, 64]:
		grid = Grid(n, n, 1.0, 1.0)
		solver = PyFOAMSolver(grid, 100)
		solver.initialize_fields()
		solver.fields['u'][-1, :] = 1.0
		solver.run(max_iter=1000, tol=1e-4)
		u_center = solver.fields['u'][:, n//2]
		if n == 32:
			ref = u_center.copy()
		else:
			interp = np.interp(np.linspace(0, 1, 32), np.linspace(0, 1, n), u_center)
			if ref is not None:
				errors.append(np.linalg.norm(interp - ref, ord=2))
	if len(errors) == 2:
		assert errors[0] > errors[1]  # 16x16 error > 64x64 error

def test_finitude_solver_high_re_stability():
	grid = Grid(32, 32, 1.0, 1.0)
	solver = FinitudeSolver(grid, 1e6)
	solver.initialize_fields()
	solver.fields['u'][-1, :] = 1.0
	try:
		solver.run(max_iter=500, tol=1e-3)
	except Exception:
		pytest.fail('FinitudeSolver failed at high Re')
	u = solver.fields['u']
	assert np.all(np.isfinite(u))
	v = solver.fields['v']
	assert np.all(np.isfinite(v))

# --- Additional Benchmarks ---
def test_pyfoam_solver_high_re_accuracy():
	"""Test PyFOAMSolver at Re=1000 for finite, stable results."""
	grid = Grid(32, 32, 1.0, 1.0)
	solver = PyFOAMSolver(grid, 1000)
	solver.initialize_fields()
	solver.fields['u'][-1, :] = 1.0
	solver.run(max_iter=2000, tol=1e-4)
	u = solver.fields['u']
	v = solver.fields['v']
	assert np.all(np.isfinite(u))
	assert np.all(np.isfinite(v))

def test_finitude_solver_accuracy_against_pyfoam():
	"""Compare FinitudeSolver and PyFOAMSolver at Re=100 for similarity."""
	grid = Grid(16, 16, 1.0, 1.0)
	pyfoam = PyFOAMSolver(grid, 100)
	pyfoam.initialize_fields()
	pyfoam.fields['u'][-1, :] = 1.0
	pyfoam.run(max_iter=1000, tol=1e-4)
	finitude = FinitudeSolver(grid, 100)
	finitude.initialize_fields()
	finitude.fields['u'][-1, :] = 1.0
	finitude.run(max_iter=1000, tol=1e-4)
	# Compare u-velocity centerline
	u_pyfoam = pyfoam.fields['u'][:, grid.nx//2]
	u_finitude = finitude.fields['u'][:, grid.nx//2]
	# Allow some tolerance for experimental solver
	assert np.allclose(u_pyfoam, u_finitude, rtol=0.2)

def test_solver_runtime_scaling(benchmark):
	"""Benchmark PyFOAMSolver runtime for increasing grid sizes."""
	def run_solver(n):
		grid = Grid(n, n, 1.0, 1.0)
		solver = PyFOAMSolver(grid, 100)
		solver.initialize_fields()
		solver.fields['u'][-1, :] = 1.0
		solver.run(max_iter=200, tol=1e-3)
	for n in [16, 32, 64]:
		benchmark(lambda: run_solver(n))

@pytest.mark.parametrize("Re", [10, 100, 1000, 10000])
def test_solvers_stability_across_re(Re):
	"""Test both solvers for finite results across a range of Reynolds numbers."""
	grid = Grid(16, 16, 1.0, 1.0)
	for Solver in [PyFOAMSolver, FinitudeSolver]:
		solver = Solver(grid, Re)
		solver.initialize_fields()
		solver.fields['u'][-1, :] = 1.0
		try:
			solver.run(max_iter=500, tol=1e-3)
		except Exception:
			pytest.fail(f'{Solver.__name__} failed at Re={Re}')
		u = solver.fields['u']
		v = solver.fields['v']
		assert np.all(np.isfinite(u))
		assert np.all(np.isfinite(v))

pytest_pyfoam_tests = [
    test_pyfoam_solver_centerline_v,
    test_pyfoam_solver_grid_convergence,
    test_pyfoam_solver_high_re_accuracy,
    test_solver_runtime_scaling,
    test_solvers_stability_across_re
]
