from __future__ import annotations

import scipy.sparse as sp


def build_pressure_matrix(nx: int, ny: int):
    """Build and return the sparse pressure matrix (SPD 5-point Laplacian).

    This pressure matrix is used in the projection step solving the Poisson
    system for pressure. Index mapping: k = j*nx + i for 0<=i<nx, 0<=j<ny
    ensuring reshape(ny,nx). Stencil: 4 on center, -1 on direct neighbors.
    """
    N = nx * ny
    data = []
    rows = []
    cols = []
    def idx(i,j):
        return j*nx + i
    for j in range(ny):
        for i in range(nx):
            k = idx(i,j)
            rows.append(k); cols.append(k); data.append(4.0)
            if i>0:
                rows.append(k); cols.append(idx(i-1,j)); data.append(-1.0)
            if i<nx-1:
                rows.append(k); cols.append(idx(i+1,j)); data.append(-1.0)
            if j>0:
                rows.append(k); cols.append(idx(i,j-1)); data.append(-1.0)
            if j<ny-1:
                rows.append(k); cols.append(idx(i,j+1)); data.append(-1.0)
    return sp.csr_matrix((data,(rows,cols)), shape=(N,N))
