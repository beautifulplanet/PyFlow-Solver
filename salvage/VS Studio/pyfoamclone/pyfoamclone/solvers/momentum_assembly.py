from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags


def build_stencil(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0, diffusion: float = 1.0):
    """Construct a simple 5-point Laplacian diffusion operator.

    Replaces prior identity placeholder so residual evolution reflects
    smoothing dynamics. Still omits advection & pressure coupling.
    """
    N = nx * ny
    main = -2.0 * diffusion * (1.0/dx**2 + 1.0/dy**2) * np.ones(N)
    off_x = diffusion / dx**2 * np.ones(N-1)
    off_y = diffusion / dy**2 * np.ones(N-nx)
    # Zero out row transitions in x-direction
    for j in range(1, ny):
        off_x[j*nx - 1] = 0.0
    return diags(
        [main, off_x, off_x, off_y, off_y],
        [0, -1, 1, -nx, nx], shape=(N, N)
    )


def apply_bc(vec: np.ndarray) -> None:  # pragma: no cover - noop prototype
    # Placeholder: real implementation would modify ghost cells / modify rows in matrix
    return None


def add_source_terms(vec: np.ndarray) -> None:  # pragma: no cover - noop prototype
    return None
