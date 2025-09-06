from __future__ import annotations

import numpy as np
import scipy.sparse as sp


import scipy.sparse.linalg as spla

def jacobi_preconditioner(A: sp.spmatrix):
    """
    Create a Jacobi (diagonal) preconditioner as a LinearOperator for use with iterative solvers.
    Args:
        A: Sparse matrix (scipy.sparse matrix)
    Returns:
        M: scipy.sparse.linalg.LinearOperator representing the Jacobi preconditioner
    """
    D = A.diagonal()
    invD = np.where(D != 0, 1.0 / D, 0.0)
    def mv(x):
        return invD * x
    return spla.LinearOperator(A.shape, matvec=mv, dtype=A.dtype)
