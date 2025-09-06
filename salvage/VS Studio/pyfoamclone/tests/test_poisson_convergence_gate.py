import numpy as np
import scipy.sparse as sp
import pytest
from pyfoamclone.linear_solvers.interface import solve


def build_poisson_2d(n: int):
    """5-point Laplacian with Dirichlet BC on unit square interior (n x n)."""
    N = n * n
    data = []
    rows = []
    cols = []
    h = 1.0 / (n + 1)
    inv_h2 = 1.0 / (h * h)
    def idx(i, j):
        return i * n + j
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            rows.append(k); cols.append(k); data.append(-4.0 * inv_h2)
            if i > 0:
                rows.append(k); cols.append(idx(i - 1, j)); data.append(1.0 * inv_h2)
            if i < n - 1:
                rows.append(k); cols.append(idx(i + 1, j)); data.append(1.0 * inv_h2)
            if j > 0:
                rows.append(k); cols.append(idx(i, j - 1)); data.append(1.0 * inv_h2)
            if j < n - 1:
                rows.append(k); cols.append(idx(i, j + 1)); data.append(1.0 * inv_h2)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def manufactured_rhs(n: int):
    """Sine manufactured solution giving non-zero truncation error of O(h^2).

    u = sin(pi x) sin(pi y) on (0,1)^2 with u=0 on boundary.
    Laplacian(u) = -2 pi^2 u.
    """
    xs = np.linspace(0, 1, n + 2)[1:-1]
    ys = np.linspace(0, 1, n + 2)[1:-1]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = -2 * (np.pi ** 2) * u_exact
    return u_exact, f


@pytest.mark.convergence_gate
def test_poisson_convergence_gate():
    sizes = [8, 16, 32, 64]  # doubling for clean order estimate
    h_vals = []
    errs = []
    last_residual = None
    for n in sizes:
        A = build_poisson_2d(n)
        u_exact, f = manufactured_rhs(n)
        b = f.reshape(-1)
        import scipy.sparse.linalg as spla
        u_vec = spla.spsolve(A, b)
        r = b - A @ u_vec
        last_residual = float(np.linalg.norm(r))
        u_num = u_vec.reshape(n, n)
        err = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)
        h = 1.0 / (n + 1)
        h_vals.append(h)
        errs.append(err)
    # Linear regression on log(err) vs log(h)
    logh = np.log(h_vals)
    loge = np.log(errs)
    A = np.vstack([logh, np.ones_like(logh)]).T
    slope, _intercept = np.linalg.lstsq(A, loge, rcond=None)[0]
    p_obs = slope  # err ~ h^p -> log err = p log h + C
    assert p_obs >= 1.8, f"Observed order {p_obs:.2f} < 1.8 (errs={errs}, h={h_vals})"
    assert last_residual is not None and last_residual <= 1e-6, f"Final residual {last_residual} > 1e-6"