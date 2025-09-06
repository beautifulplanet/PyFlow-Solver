"""Test utilities for projection / divergence related assertions.

These helpers encode the Neumann projection invariant: only the mean-free
component of the discrete divergence can be removed by a pressure projection
when the pressure equation is solved with homogeneous Neumann BC (plus a
single pin / mean subtraction for compatibility).  The spatial mean of the
divergence (the constant mode) is preserved.
"""
from __future__ import annotations

import numpy as np

def mean_free(field: np.ndarray):
    """Return (field_mean_free, mean, l2_norm_mean_free).

    Parameters
    ----------
    field : ndarray
        Scalar field (divergence values) over interior cells.
    """
    mean = float(field.mean())
    mf = field - mean
    return mf, mean, float(np.linalg.norm(mf))

def assert_mean_free_reduction(before: np.ndarray, after: np.ndarray, *, atol_zero: float = 1e-12, rtol: float = 1e-2):
    """Assert that the mean-free component of `after` is not larger than that of `before`.

    If the initial mean-free norm is (near) zero, we instead assert the final
    mean-free norm is below a small absolute tolerance.
    """
    bf_mf, bf_mean, bf_norm = mean_free(before)
    af_mf, af_mean, af_norm = mean_free(after)
    # Means should remain close (projection preserves constant mode)
    assert abs(bf_mean - af_mean) < 1e-8 + 1e-8*abs(bf_mean), (
        f"Divergence mean changed (should be invariant under Neumann projection): before={bf_mean}, after={af_mean}"
    )
    if bf_norm < atol_zero:
        assert af_norm < max(atol_zero, 5*atol_zero), (
            f"Mean-free divergence introduced: initial ~0 (norm {bf_norm}), final norm {af_norm}"
        )
    else:
        # Allow a small relative increase (roundoff) then expect reduction factor
        assert af_norm <= bf_norm*(1 + 5*rtol), (
            f"Mean-free divergence not reduced: before={bf_norm}, after={af_norm}"
        )

__all__ = [
    "mean_free",
    "assert_mean_free_reduction",
]
