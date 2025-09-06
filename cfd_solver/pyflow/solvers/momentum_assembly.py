from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def build_stencil(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0, diffusion: float = 1.0):
    """Construct a simple 5-point Laplacian diffusion operator (CSR)."""
    N = nx * ny
    main = -2.0 * diffusion * (1.0/dx**2 + 1.0/dy**2) * np.ones(N)
    off_x = diffusion / dx**2 * np.ones(N-1)
    off_y = diffusion / dy**2 * np.ones(N-nx)
    for j in range(1, ny):
        off_x[j*nx - 1] = 0.0
    A = sp.diags([main, off_x, off_x, off_y, off_y], [0, -1, 1, -nx, nx], shape=(N, N))  # type: ignore[arg-type]
    return A.tocsr()

def apply_bc(vec: np.ndarray) -> None:  # placeholder
    return None

def add_source_terms(vec: np.ndarray) -> None:  # placeholder
    return None

__all__ = ['build_stencil', 'apply_bc', 'add_source_terms']
