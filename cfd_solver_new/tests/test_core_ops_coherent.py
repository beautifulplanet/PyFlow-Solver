import numpy as np
import pytest
from pyflow.numerics.core_ops_coherent import (
    gradient_coherent,
    divergence_coherent,
    laplacian_coherent,
    coherence_diagnostics,
)

@pytest.mark.parametrize("nx,ny", [(8,8), (16,12), (5,9)])
def test_laplacian_equals_divergence_gradient(nx, ny):
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    rng = np.random.default_rng(42)
    p = rng.standard_normal((ny, nx))
    dpdx, dpdy = gradient_coherent(p, dx, dy)
    divgrad = divergence_coherent(dpdx, dpdy, dx, dy)
    lap = laplacian_coherent(p, dx, dy)
    diff = lap - divgrad
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(lap) or 1.0)
    assert rel_l2 < 5e-13, f"Operator mismatch rel_l2={rel_l2:.3e}"
    assert np.max(np.abs(diff)) < 1e-12

@pytest.mark.parametrize("nx,ny", [(8,8), (11,7)])
def test_laplacian_constant_field_zero(nx, ny):
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    c = np.full((ny, nx), 3.14)
    lap = laplacian_coherent(c, dx, dy)
    assert np.allclose(lap, 0.0, atol=1e-14)

@pytest.mark.parametrize("nx,ny", [(8,8), (16,16)])
def test_divergence_of_gradient_mean_free(nx, ny):
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    rng = np.random.default_rng(7)
    p = rng.standard_normal((ny, nx))
    dpdx, dpdy = gradient_coherent(p, dx, dy)
    divgrad = divergence_coherent(dpdx, dpdy, dx, dy)
    # For Neumann boundaries, integral of Laplacian (divgrad) should be near zero
    mean_val = float(np.mean(divgrad))
    assert abs(mean_val) < 5e-13


def test_coherence_diagnostics_summary():
    d = coherence_diagnostics(12, 10, 1/11, 1/9, seed=3)
    assert d.l2_rel_diff < 5e-13
    assert d.constant_laplacian_max < 1e-14
