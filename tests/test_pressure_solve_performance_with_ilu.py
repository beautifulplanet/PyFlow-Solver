import time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import pytest
from pyfoamclone.linear_solvers.preconditioners import jacobi_preconditioner, ilu_preconditioner

def build_poisson(n):
    # 2D 5-point Laplacian on unit square (Dirichlet)
    N = n*n
    main = np.full(N, 4.0)
    off1 = np.full(N-1, -1.0)
    offn = np.full(N-n, -1.0)
    for k in range(1, n):
        off1[k*n-1] = 0  # block row boundary
    # Use positional arguments for SciPy compatibility (avoid type checker confusion)
    A = sp.diags([main, off1, off1, offn, offn], [0, -1, 1, -n, n], format='csc')  # type: ignore[arg-type]
    return A

def run_solver(A, M=None):
    b = np.ones(A.shape[0])
    iters = {'k':0}
    def cb(x):
        iters['k'] += 1
    start = time.time()
    # Use BiCGSTAB to accommodate nonsymmetric ILU preconditioner robustly
    x, info = spla.bicgstab(A, b, M=M, atol=0, rtol=1e-8, maxiter=5000, callback=cb)
    dt = time.time() - start
    return iters['k'] if iters['k']>0 else (info if info>0 else 5000), dt

@pytest.mark.slow
def test_pressure_solve_performance_with_ilu():
    n = 129
    A = build_poisson(n)
    # No preconditioner
    it_none, t_none = run_solver(A, M=None)
    # Jacobi
    M_j = jacobi_preconditioner(A)
    it_j, t_j = run_solver(A, M=M_j)
    # ILU
    M_i = ilu_preconditioner(A, fill_factor=5, drop_tol=1e-4)
    it_i, t_i = run_solver(A, M=M_i)
    assert it_i <= it_j + 1, f"ILU iterations not <= Jacobi ({it_i} > {it_j})"
    assert it_j <= it_none + 1, f"Jacobi not better than none ({it_j} > {it_none})"
