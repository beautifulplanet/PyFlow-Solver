import numpy as np
import pytest
import scipy.sparse as sp
from pyfoamclone.linear_solvers.preconditioners import jacobi_preconditioner
from pyfoamclone.solvers.pressure_solver import assemble_negative_laplacian, solve_pressure_poisson
from pyfoamclone.core.ghost_fields import allocate_state

@pytest.mark.perf
def test_pressure_solve_performance_no_preconditioner():
    nx = ny = 129
    dx = dy = 1.0 / (nx - 1)
    A = assemble_negative_laplacian(nx, ny, dx, dy)
    # Manufactured solution: p = sin(pi x) sin(pi y)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    p_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    b = (2 * np.pi**2) * p_true.reshape(-1)  # -Laplace(p_true)
    b -= np.mean(b)  # ensure sum(b) == 0 for compatibility
    b[0] = 0.0  # reference cell
    class Cfg: lin_tol=1e-10; lin_maxiter=1000
    state = allocate_state(nx, ny)
    state.meta['A_press'] = A
    state.meta['A_press_shape'] = (nx, ny)
    state.meta['A_press_dx'] = dx
    state.meta['A_press_dy'] = dy
    print(f"A shape: {A.shape}, nnz: {A.nnz}, b norm: {np.linalg.norm(b)}, b[0]: {b[0]}, sum(b): {np.sum(b)}")
    _, diag = solve_pressure_poisson(state, 1.0, dx, dy, Cfg, preconditioner=None, rhs_override=b)
    print(f"CG diagnostics (no preconditioner): {diag}")
    iters = diag['Rp_iterations']
    assert iters > 0
    return iters

@pytest.mark.perf
def test_pressure_solve_performance_with_jacobi():
    nx = ny = 129
    dx = dy = 1.0 / (nx - 1)
    A = assemble_negative_laplacian(nx, ny, dx, dy)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    p_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
    b = (2 * np.pi**2) * p_true.reshape(-1)
    b -= np.mean(b)
    b[0] = 0.0
    class Cfg: lin_tol=1e-10; lin_maxiter=1000
    state = allocate_state(nx, ny)
    state.meta['A_press'] = A
    state.meta['A_press_shape'] = (nx, ny)
    state.meta['A_press_dx'] = dx
    state.meta['A_press_dy'] = dy
    jacobi = jacobi_preconditioner(A)
    print(f"A shape: {A.shape}, nnz: {A.nnz}, b norm: {np.linalg.norm(b)}, b[0]: {b[0]}, sum(b): {np.sum(b)}")
    _, diag = solve_pressure_poisson(state, 1.0, dx, dy, Cfg, preconditioner=jacobi, rhs_override=b)
    print(f"CG diagnostics (Jacobi): {diag}")
    iters = diag['Rp_iterations']
    assert iters > 0
    return iters

def test_jacobi_preconditioner_reduces_iterations():
    iters_no_pc = test_pressure_solve_performance_no_preconditioner()
    iters_jacobi = test_pressure_solve_performance_with_jacobi()
    # Jacobi should reduce iteration count by at least 2x
    assert iters_jacobi < iters_no_pc * 0.5, f"Jacobi did not reduce iterations enough: {iters_no_pc} vs {iters_jacobi}"
