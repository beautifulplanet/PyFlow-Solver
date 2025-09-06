import numpy as np
from pyfoamclone.numerics.stencil.coeffs import laplacian_coeffs, apply_laplacian
from pyfoamclone.numerics.interp import central, upwind
from pyfoamclone.numerics.time_step import compute_dt
from pyfoamclone.residuals.manager import ResidualManager


def test_laplacian_coeffs():
    cxm, cxp, cym, cyp, cc = laplacian_coeffs(0.5, 0.25)
    assert cc < 0 and abs(cxm - cxp) < 1e-12 and abs(cym - cyp) < 1e-12


def test_apply_laplacian_quadratic():
    nx, ny = 16, 16
    dx = dy = 1.0 / nx
    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = X**2 + Y**2
    lap = apply_laplacian(f, dx, dy)
    interior = lap[2:-2, 2:-2]
    assert np.allclose(interior.mean(), 4, rtol=0.2)


def test_interp_flux_balance():
    a = np.ones(5)
    b = np.ones(5) * 3
    c = central(a, b)
    assert np.allclose(c, 2)
    sign = np.array([1, -1, 1, -1, 1])
    up = upwind(a, b, sign)
    assert up[1] == b[1] and up[0] == a[0]


def test_compute_dt_growth_cap():
    dt = compute_dt(0.5, current_cfl=0.2, prev_dt=0.01, max_growth=1.1)
    # target factor would be 2.5 but limited to 1.1
    assert abs(dt - 0.011) < 1e-12


def test_residual_tracker_plateau():
    rt = ResidualManager()
    # create near-plateau sequence (slow decline)
    vals = [1.0 - 0.0005 * i for i in range(100)]
    for v in vals:
        rt.add("u", v)
    assert rt.plateau_detect("u", window=80, threshold=-0.02) is True
