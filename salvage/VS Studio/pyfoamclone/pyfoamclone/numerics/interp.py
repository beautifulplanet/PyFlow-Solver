from __future__ import annotations

import numpy as np
from typing import Literal


def central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


def upwind(a: np.ndarray, b: np.ndarray, flux_sign: np.ndarray) -> np.ndarray:
    return np.where(flux_sign >= 0, a, b)


def rhie_chow(u_f: np.ndarray, p: np.ndarray, d: float) -> np.ndarray:
    """Very light placeholder Rhie–Chow style smoothing.

    u_f: face velocities (approx)
    p: cell-centered pressure
    d: damping factor (0..1)
    """
    # Smooth using neighboring average (1D illustrative)
    smooth = 0.5 * (u_f[..., :-1] + u_f[..., 1:])
    corr = np.zeros_like(u_f)
    corr[..., 1:-1] = smooth
    return (1 - d) * u_f + d * corr
