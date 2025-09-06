from __future__ import annotations
"""Matrix-free numerical operators for PyFlow.

Phase 2 (Isolated R&D) deliverable:

This module provides a matrix-free Laplacian constructed compositionally as
    L(p) = div( grad(p) )
using the project's existing discrete gradient and divergence operators
(central differences in the interior; one‑sided on boundaries).

Goals:
1. Supply a reference implementation of the Laplacian via operator composition.
2. Provide a way to assemble the *exact* sparse matrix corresponding to this
   composition for small grids (for verification only) without relying on the
   legacy pressure matrix whose boundary treatment differs.

Notes:
- No pin / reference pressure is applied here; the null space (constants) is
  preserved for validation (null space test).
- All functions are pure: no mutation of solver State objects or global data.
- Intended for unit tests and later controlled integration.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from .fluid_ops import gradient, divergence

Array2D = np.ndarray

@dataclass(slots=True)
class OperatorComparisonResult:
    rel_error_l2: float
    abs_error_max: float
    lhs_norm: float
    rhs_norm: float


def laplacian_matrix_free(p: Array2D, dx: float, dy: float) -> Array2D:
    """Compute L(p) = div(grad(p)) using existing discrete ops.

    Args:
        p: 2D scalar field (ny, nx)
        dx, dy: grid spacings
    Returns:
        2D array of Laplacian values with same shape.
    """
    dpdx, dpdy = gradient(p, dx, dy)
    lap = divergence(dpdx, dpdy, dx, dy)
    return lap


def build_laplacian_matrix_from_ops(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """Assemble the exact sparse matrix representation of the composition L = div(grad).

    This constructs columns by applying the operator to each basis vector e_k.
    For validation only (O(N^2)), feasible for small N (<= 256 cells).
    Boundary handling matches the composed operators exactly.

    Args:
        nx, ny: grid resolution
        dx, dy: spacings
    Returns:
        CSR sparse matrix of shape (N, N), N = nx*ny
    """
    N = nx * ny
    A = sp.lil_matrix((N, N))
    basis = np.zeros((ny, nx))
    for k in range(N):
        j = k // nx
        i = k - j * nx
        basis[j, i] = 1.0
        col_field = laplacian_matrix_free(basis, dx, dy)
        A[:, k] = col_field.reshape(-1)
        basis[j, i] = 0.0  # reset
    return sp.csr_matrix(A)


def compare_operator_with_matrix(p: Array2D, A: sp.csr_matrix, dx: float, dy: float) -> OperatorComparisonResult:
    """Compare matrix-free Laplacian with explicit matrix product.

    Args:
        p: scalar field (ny, nx)
        A: sparse matrix built by build_laplacian_matrix_from_ops (shape N,N)
        dx, dy: spacings
    Returns:
        OperatorComparisonResult with relative L2 error and max abs error.
    """
    lap_free = laplacian_matrix_free(p, dx, dy).reshape(-1)
    lap_mat = (A @ p.reshape(-1))
    diff = lap_free - lap_mat
    lhs_norm = np.linalg.norm(lap_free)
    rhs_norm = np.linalg.norm(lap_mat)
    rel = np.linalg.norm(diff) / (rhs_norm if rhs_norm else 1.0)
    return OperatorComparisonResult(rel_error_l2=float(rel),
                                    abs_error_max=float(np.max(np.abs(diff))),
                                    lhs_norm=float(lhs_norm),
                                    rhs_norm=float(rhs_norm))


# ---------------------------------------------------------------------------
# Correct 5-point Laplacian (direct stencil) implementation
# ---------------------------------------------------------------------------
def laplacian_matrix_free_5point(p: Array2D, dx: float, dy: float) -> Array2D:
    """Apply the standard 5-point Laplacian directly.

    Interior stencil (ny,nx):
        (p[i+1,j] - 2p[i,j] + p[i-1,j]) / dx^2 + (p[i,j+1] - 2p[i,j] + p[i,j-1]) / dy^2

    Boundary handling: replicate nearest interior second derivative (same policy
    as existing laplacian() function in fluid_ops for continuity of tests).
    """
    ny, nx = p.shape
    lap = np.zeros_like(p)
    if nx > 2 and ny > 2:
        lap[1:-1,1:-1] = (
            (p[1:-1,2:] - 2.0*p[1:-1,1:-1] + p[1:-1,0:-2]) / (dx*dx) +
            (p[2:,1:-1] - 2.0*p[1:-1,1:-1] + p[0:-2,1:-1]) / (dy*dy)
        )
    # Boundary replication (crude, consistent with earlier design)
    if ny > 1:
        lap[0,:] = lap[1,:]
        lap[-1,:] = lap[-2,:]
    if nx > 1:
        lap[:,0] = lap[:,1]
        lap[:,-1] = lap[:,-2]
    return lap

def laplacian_matrix_free_5point_neumann(p: Array2D, dx: float, dy: float) -> Array2D:
    """5-point Laplacian with homogeneous Neumann boundaries (zero normal derivative).

    Implements reflective ghost cell logic: p_{-1} = p_0, p_{nx} = p_{nx-1}, etc.
    Interior: standard second differences. Boundaries: one-sided derived from reflection.
    """
    ny, nx = p.shape
    lap = np.zeros_like(p)
    if nx > 2 and ny > 2:
        # Interior
        lap[1:-1,1:-1] = (
            (p[1:-1,2:] - 2.0*p[1:-1,1:-1] + p[1:-1,0:-2]) / (dx*dx) +
            (p[2:,1:-1] - 2.0*p[1:-1,1:-1] + p[0:-2,1:-1]) / (dy*dy)
        )
    # Left / Right (excluding corners)
    if nx > 1:
        if nx > 2:
            # i=0
            lap[1:-1,0] = (p[1:-1,1] - p[1:-1,0]) / (dx*dx) + (
                (p[2:,0] - 2.0*p[1:-1,0] + p[0:-2,0]) / (dy*dy)
            )
            # i=nx-1
            lap[1:-1,-1] = (p[1:-1,-2] - p[1:-1,-1]) / (dx*dx) + (
                (p[2:,-1] - 2.0*p[1:-1,-1] + p[0:-2,-1]) / (dy*dy)
            )
    # Top / Bottom (excluding corners)
    if ny > 1:
        if ny > 2:
            # j=0
            lap[0,1:-1] = (p[0,2:] - 2.0*p[0,1:-1] + p[0,0:-2]) / (dx*dx) + (
                (p[1,1:-1] - p[0,1:-1]) / (dy*dy)
            )
            # j=ny-1
            lap[-1,1:-1] = (p[-1,2:] - 2.0*p[-1,1:-1] + p[-1,0:-2]) / (dx*dx) + (
                (p[-2,1:-1] - p[-1,1:-1]) / (dy*dy)
            )
    # Corners j=0,i=0 etc.
    if nx > 1 and ny > 1:
        # (0,0)
        lap[0,0] = (p[0,1] - p[0,0]) / (dx*dx) + (p[1,0] - p[0,0]) / (dy*dy)
        # (0,nx-1)
        lap[0,-1] = (p[0,-2] - p[0,-1]) / (dx*dx) + (p[1,-1] - p[0,-1]) / (dy*dy)
        # (ny-1,0)
        lap[-1,0] = (p[-1,1] - p[-1,0]) / (dx*dx) + (p[-2,0] - p[-1,0]) / (dy*dy)
        # (ny-1,nx-1)
        lap[-1,-1] = (p[-1,-2] - p[-1,-1]) / (dx*dx) + (p[-2,-1] - p[-1,-1]) / (dy*dy)
    return lap

def build_5point_laplacian_matrix(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """Assemble sparse matrix for the 5-point Laplacian with boundary replication.

    Boundary rows approximate by copying nearest interior second derivative result:
    implemented here by mirroring interior stencil onto boundary using the same
    replication logic as laplacian_matrix_free_5point.
    """
    N = nx * ny
    A = sp.lil_matrix((N, N))
    basis = np.zeros((ny, nx))
    for k in range(N):
        j = k // nx
        i = k - j * nx
        basis[j, i] = 1.0
        col_field = laplacian_matrix_free_5point(basis, dx, dy)
        A[:, k] = col_field.reshape(-1)
        basis[j, i] = 0.0
    return sp.csr_matrix(A)


__all__ = [
    'laplacian_matrix_free',
    'build_laplacian_matrix_from_ops',
    'compare_operator_with_matrix',
    'laplacian_matrix_free_5point',
    'build_5point_laplacian_matrix',
    'laplacian_matrix_free_5point_neumann',
    'OperatorComparisonResult'
]
