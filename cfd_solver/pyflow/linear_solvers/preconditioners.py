from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any

MatrixLike = Any  # broaden for scipy csr_array compatibility

def _to_csr(A: MatrixLike) -> sp.csr_matrix:
    return A if isinstance(A, sp.csr_matrix) else sp.csr_matrix(A)

def jacobi_preconditioner(A: MatrixLike) -> spla.LinearOperator:  # type: ignore[name-defined]
    A_csr = _to_csr(A)
    D: np.ndarray = A_csr.diagonal()  # ensure ndarray
    invD = np.where(D != 0, 1.0 / D, 0.0)
    def mv(x):  # pragma: no cover - simple lambda
        return invD * x
    return spla.LinearOperator(shape=A_csr.shape, matvec=mv)  # type: ignore[arg-type]

def ilu_preconditioner(A: MatrixLike, fill_factor: float = 10.0, drop_tol: float = 1e-5) -> spla.LinearOperator:  # type: ignore[name-defined]
    A_csc = A if isinstance(A, sp.csc_matrix) else _to_csr(A).tocsc()
    ilu = spla.spilu(A_csc, fill_factor=fill_factor, drop_tol=drop_tol)
    def mv(x):  # pragma: no cover
        return ilu.solve(x)
    return spla.LinearOperator(shape=A_csc.shape, matvec=mv)  # type: ignore[arg-type]

__all__ = ['jacobi_preconditioner', 'ilu_preconditioner']
