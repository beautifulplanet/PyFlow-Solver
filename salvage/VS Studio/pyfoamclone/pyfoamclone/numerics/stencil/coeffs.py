from __future__ import annotations

import os
import numpy as np
from typing import Tuple

_USE_NUMBA = os.environ.get('PYFOAMCLONE_USE_NUMBA', '0') == '1'
try:  # optional
    if _USE_NUMBA:
        from numba import njit  # type: ignore
    else:  # pragma: no cover
        def njit(*args, **kwargs):  # type: ignore
            def wrap(f):
                return f
            return wrap
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap


def laplacian_coeffs(dx: float, dy: float) -> Tuple[float, float, float, float, float]:
    """Return 5-point Laplacian stencil coefficients (cxm, cxp, cym, cyp, cc).

    For uniform grid, classic 2D Laplacian discretization:
    (phi_{i-1,j} -2 phi_{i,j} + phi_{i+1,j})/dx^2 + (phi_{i,j-1} -2 phi_{i,j} + phi_{i,j+1})/dy^2
    """
    cxm = 1.0 / dx ** 2
    cxp = 1.0 / dx ** 2
    cym = 1.0 / dy ** 2
    cyp = 1.0 / dy ** 2
    cc = -2.0 * (1.0 / dx ** 2 + 1.0 / dy ** 2)
    return cxm, cxp, cym, cyp, cc


@njit(cache=True)
def _laplacian_core(field: np.ndarray, out: np.ndarray, cxm: float, cxp: float, cym: float, cyp: float, cc: float):  # type: ignore
    ny, nx = field.shape
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            out[i, j] = (
                cxm * field[i, j - 1]
                + cxp * field[i, j + 1]
                + cym * field[i - 1, j]
                + cyp * field[i + 1, j]
                + cc * field[i, j]
            )


def apply_laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    cxm, cxp, cym, cyp, cc = laplacian_coeffs(dx, dy)
    out = np.zeros_like(field)
    if _USE_NUMBA:
        _laplacian_core(field, out, cxm, cxp, cym, cyp, cc)
        return out
    # vectorized fallback
    out[1:-1, 1:-1] = (
        cxm * field[1:-1, 0:-2]
        + cxp * field[1:-1, 2:]
        + cym * field[0:-2, 1:-1]
        + cyp * field[2:, 1:-1]
        + cc * field[1:-1, 1:-1]
    )
    return out

