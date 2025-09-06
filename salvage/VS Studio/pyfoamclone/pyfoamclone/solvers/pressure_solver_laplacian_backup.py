from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from ..core.ghost_fields import interior_view, State
from ..numerics.fluid_ops import divergence, gradient
from ..linear_solvers.interface import solve

def assemble_negative_laplacian(nx: int, ny: int, dx: float, dy: float):
    """Backup of Laplacian assembly as of August 31, 2025."""
    N = nx * ny
    rows = []
    cols = []
    data = []
    idx = lambda i,j: j*nx + i
    invdx2 = 1.0 / (dx*dx)
    invdy2 = 1.0 / (dy*dy)
    center = 2.0*(invdx2 + invdy2)
    for j in range(ny):
        for i in range(nx):
            k = idx(i,j)
            rows.append(k); cols.append(k); data.append(center)
            if i>0: rows.append(k); cols.append(idx(i-1,j)); data.append(-invdx2)
            if i<nx-1: rows.append(k); cols.append(idx(i+1,j)); data.append(-invdx2)
            if j>0: rows.append(k); cols.append(idx(i,j-1)); data.append(-invdy2)
            if j<ny-1: rows.append(k); cols.append(idx(i,j+1)); data.append(-invdy2)
    A = sp.csr_matrix((data,(rows,cols)), shape=(N,N)).tolil()
    # Reference cell to remove nullspace (p=0 at (0,0))
    A[0,:] = 0.0; A[0,0] = 1.0
    return A.tocsr()
