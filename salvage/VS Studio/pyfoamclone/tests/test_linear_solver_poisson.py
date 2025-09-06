import numpy as np
import scipy.sparse as sp
from pyfoamclone.linear_solvers.interface import solve


def build_poisson_2d(n):
    # classic 5-point Laplacian on unit square interior points n x n (Dirichlet 0)
    N = n * n
    data = []
    rows = []
    cols = []
    def idx(i,j):
        return i*n + j
    for i in range(n):
        for j in range(n):
            k = idx(i,j)
            rows.append(k); cols.append(k); data.append(-4.0)
            if i>0:
                rows.append(k); cols.append(idx(i-1,j)); data.append(1.0)
            if i<n-1:
                rows.append(k); cols.append(idx(i+1,j)); data.append(1.0)
            if j>0:
                rows.append(k); cols.append(idx(i,j-1)); data.append(1.0)
            if j<n-1:
                rows.append(k); cols.append(idx(i,j+1)); data.append(1.0)
    A = sp.csr_matrix((data,(rows,cols)), shape=(N,N))
    return A


def test_poisson_manufactured():
    n = 12
    A = build_poisson_2d(n)
    # manufactured solution u = sin(pi x) sin(pi y); Laplacian(u) = -2 pi^2 u
    xs = np.linspace(0,1,n+2)[1:-1]
    ys = np.linspace(0,1,n+2)[1:-1]
    X,Y = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(np.pi*X)*np.sin(np.pi*Y)
    f = -2*(np.pi**2)*u_exact
    b = f.reshape(-1)
    res = solve(A, b, method='bicgstab', tol=1e-6, maxiter=200)
    u_num = res.x.reshape(n,n)
    # scale difference (matrix corresponds to discrete Laplacian without /h^2 scaling applied)
    # we just check proportional similarity up to scalar factor: correlation close to 1
    corr = np.corrcoef(u_exact.reshape(-1), u_num.reshape(-1))[0,1]
    assert corr > 0.999
    assert res.converged is True
