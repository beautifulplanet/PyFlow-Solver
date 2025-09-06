from __future__ import annotations
"""Legacy preconditioners (kept for backward compatibility with salvage tests).

Adjusted for stricter type checkers (Pylance) by:
- Restricting accepted matrix types to concrete sparse matrix subclasses
- Using positional args for LinearOperator (some SciPy stubs lack "matvec" kw)
- Adding targeted type: ignore comments where SciPy stubs are incomplete
"""

from typing import Union, Callable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SparseMatrix = Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix, sp.dia_matrix]


def _ensure_csr(A: sp.spmatrix) -> sp.csr_matrix:
    return A if isinstance(A, sp.csr_matrix) else A.tocsr()  # type: ignore[no-any-return]


def jacobi_preconditioner(A: SparseMatrix) -> spla.LinearOperator:
    A_csr = _ensure_csr(A)
    D = A_csr.diagonal()  # type: ignore[attr-defined]
    # Cast to float64 explicitly for stable division semantics
    D = D.astype(float, copy=False)  # type: ignore[call-arg]
    with np.errstate(divide='ignore'):
        invD = np.where(D != 0, 1.0 / D, 0.0)

    def mv(x: np.ndarray) -> np.ndarray:
        return invD * x

    # Use positional arguments; add ignore for potential signature variance across SciPy versions.
    return spla.LinearOperator(A_csr.shape, mv, dtype=A_csr.dtype)  # type: ignore[arg-type]


def ilu_preconditioner(
    A: SparseMatrix,
    fill_factor: float = 10.0,
    drop_tol: float = 1e-5,
) -> spla.LinearOperator:
    """Construct an ILU preconditioner wrapped as LinearOperator."""

    if not sp.isspmatrix_csc(A):
        A_csc = A.tocsc()  # type: ignore[attr-defined]
    else:
        A_csc = A  # type: ignore[assignment]
    ilu = spla.spilu(A_csc, fill_factor=fill_factor, drop_tol=drop_tol)

    def mv(x: np.ndarray) -> np.ndarray:
        return ilu.solve(x)

    return spla.LinearOperator(A_csc.shape, mv, dtype=A_csc.dtype)  # type: ignore[arg-type]


__all__ = ["jacobi_preconditioner", "ilu_preconditioner"]
