from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

def _call_solver(solver_fn, A, b, x0, tol, maxiter, **kwargs):
    try:  # SciPy >=2 uses rtol
        return solver_fn(A, b, x0=x0, rtol=tol, maxiter=maxiter, **kwargs)
    except TypeError:  # SciPy <2 uses tol
        return solver_fn(A, b, x0=x0, tol=tol, maxiter=maxiter, **kwargs)

@dataclass(slots=True)
class LinearSolveResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    method: str

def solve(A, b, method: Literal['cg','bicgstab']='bicgstab', tol: float=1e-8, maxiter: int=500, M=None, **kwargs) -> LinearSolveResult:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    b = np.asarray(b)
    x0 = np.zeros_like(b)
    if method == 'cg':
        x, info = _call_solver(spla.cg, A, b, x0, tol, maxiter, M=M, **kwargs)
    else:
        x, info = _call_solver(spla.bicgstab, A, b, x0, tol, maxiter, M=M, **kwargs)
    r = b - A @ x
    rnorm = float(np.linalg.norm(r))
    bnorm = float(np.linalg.norm(b)) or 1.0
    converged = (info == 0) and (rnorm <= tol * bnorm)
    iterations = info if info > 0 else (0 if info == 0 else maxiter)
    return LinearSolveResult(x=x, converged=converged, iterations=iterations, residual_norm=rnorm, method=method)

__all__ = ['solve', 'LinearSolveResult']
