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
    # Accept only scipy.sparse.spmatrix for A
    # For scipy.sparse.spmatrix, diagonal() and dtype are always available
    D = np.asarray(A.diagonal()).flatten()  # type: ignore[attr-defined]
    invD = np.where(D != 0, 1.0 / D, 0.0)
    def mv(x):
        return invD * x
    # Only pass dtype if not already set by A
    # LinearOperator expects shape, matvec, and optionally dtype (but not twice)
    # Only pass dtype if not already set by A; LinearOperator expects shape, matvec, and optionally dtype
    return spla.LinearOperator(A.shape, mv)
