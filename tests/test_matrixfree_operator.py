from __future__ import annotations
"""Phase 2 Verification Tests: Matrix-Free Laplacian Operator

Covers:
- Equivalence: div(grad(p)) matrix-free vs explicitly assembled (composition) matrix.
- Null space: constant field maps to ~0 (within floating tolerance).

Grid choices kept modest to allow O(N^2) assembly.
"""
import numpy as np
import pytest
from pyflow.numerics.mf_ops import (
    laplacian_matrix_free,
    build_laplacian_matrix_from_ops,
    compare_operator_with_matrix,
    laplacian_matrix_free_5point,
    build_5point_laplacian_matrix,
)

@pytest.mark.parametrize("nx,ny", [(8,8), (12,10)])
def test_matrix_free_equivalence(nx: int, ny: int):
    dx = dy = 1.0
    # Smooth analytic field (non-trivial): p = sin(ax) + cos(by) + 0.1*x*y
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    a = 2.0 * np.pi / max(1.0, (nx-1)*dx)
    b = 3.0 * np.pi / max(1.0, (ny-1)*dy)
    p = np.sin(a*X) + np.cos(b*Y) + 0.1 * X * Y

    A = build_laplacian_matrix_from_ops(nx, ny, dx, dy)
    cmp = compare_operator_with_matrix(p, A, dx, dy)

    # Tight tolerances: expect near machine precision consistency
    assert cmp.rel_error_l2 < 1e-12, f"Relative L2 error too high: {cmp.rel_error_l2}"
    assert cmp.abs_error_max < 5e-12, f"Max abs error too high: {cmp.abs_error_max}"


def test_matrix_free_null_space_constant():
    nx = ny = 16
    dx = dy = 1.0
    p = np.full((ny, nx), 7.3)
    lap = laplacian_matrix_free(p, dx, dy)
    l2 = np.linalg.norm(lap)
    assert l2 < 1e-14, f"Null space not preserved for constant field (||L c||={l2})"


@pytest.mark.parametrize("nx,ny", [(8,8), (12,10)])
def test_matrix_free_5point_equivalence(nx: int, ny: int):
    dx = dy = 1.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    a = 2.0 * np.pi / max(1.0, (nx-1)*dx)
    b = 3.0 * np.pi / max(1.0, (ny-1)*dy)
    p = np.sin(a*X) + np.cos(b*Y) + 0.05 * X * Y
    A5 = build_5point_laplacian_matrix(nx, ny, dx, dy)
    lap_free = laplacian_matrix_free_5point(p, dx, dy).reshape(-1)
    lap_mat = A5 @ p.reshape(-1)
    diff = lap_free - lap_mat
    rel = np.linalg.norm(diff) / (np.linalg.norm(lap_mat) or 1.0)
    assert rel < 1e-12, f"5-point operator mismatch rel={rel}"

