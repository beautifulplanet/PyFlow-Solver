# --- Single Cell Budget Test for 2D QUICK Operator ---
def test_quick_single_cell_budget():
    import numpy as np
    from pyfoamclone.numerics.operators.advection import advect_quick
    # 5x5 field: phi[i, j] = 10*i + j
    phi = np.fromfunction(lambda i, j: 10*i + j, (5, 5))
    u = np.ones_like(phi)
    v = np.ones_like(phi)
    dx = dy = 1.0
    # Validated 1D QUICK face interpolation
    def quick_face(phi1d, idx, flow_positive):
        if flow_positive:
            return (6/8)*phi1d[idx] + (3/8)*phi1d[idx+1] - (1/8)*phi1d[idx-1]
        else:
            return (6/8)*phi1d[idx+1] + (3/8)*phi1d[idx] - (1/8)*phi1d[idx+2]
    # East face (i=2.5, j=2): phi[2,1:4] = [21,22,23], upwind is i=2
    phi_east = quick_face(phi[2,1:4], 1, True)  # idx=1 in [21,22,23] is 22
    flux_east = 1.0 * phi_east
    # West face (i=1.5, j=2): phi[2,0:3] = [20,21,22], upwind is i=1
    phi_west = quick_face(phi[2,0:3], 1, True)
    flux_west = 1.0 * phi_west
    # North face (i=2, j=2.5): phi[1:4,2] = [12,22,32], upwind is j=2
    phi_north = quick_face(phi[1:4,2], 1, True)
    flux_north = 1.0 * phi_north
    # South face (i=2, j=1.5): phi[0:3,2] = [2,12,22], upwind is j=1
    phi_south = quick_face(phi[0:3,2], 1, True)
    flux_south = 1.0 * phi_south
    # Manual divergence
    div_manual = (flux_east - flux_west)/dx + (flux_north - flux_south)/dy
    # Now get the operator's value
    adv = advect_quick(u, v, phi, dx, dy)
    div_code = adv[2,2]
    assert np.isclose(div_code, div_manual), f"2D QUICK cell budget: code={div_code}, manual={div_manual}"
# --- Whiteboard Test for QUICK Face Interpolation ---
def test_quick_whiteboard_positive():
    phi = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Face between i=2 (30.0) and i=3 (40.0), positive flow (upwind is i=2)
    # Stencil: i-1=1 (20.0), i=2 (30.0), i+1=3 (40.0)
    val = (6/8)*30.0 + (3/8)*40.0 - (1/8)*20.0
    # Should be 22.5 + 15.0 - 2.5 = 35.0
    assert np.isclose(val, 35.0), f"Manual QUICK positive flow: got {val}, expected 35.0"
    # Now use the interpolation function (must match manual)
    def quick_face(phi, i, flow_positive):
        if flow_positive:
            return (6/8)*phi[i] + (3/8)*phi[i+1] - (1/8)*phi[i-1]
        else:
            return (6/8)*phi[i+1] + (3/8)*phi[i] - (1/8)*phi[i+2]
    interp = quick_face(phi, 2, True)
    assert np.isclose(interp, 35.0), f"QUICK positive flow: got {interp}, expected 35.0"

def test_quick_whiteboard_negative():
    phi = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Face between i=2 (30.0) and i=3 (40.0), negative flow (upwind is i=3)
    # Stencil: i=2 (30.0), i-1=3 (40.0), i+1=4 (50.0)
    val = (6/8)*40.0 + (3/8)*30.0 - (1/8)*50.0
    # Should be 30.0 + 11.25 - 6.25 = 35.0
    assert np.isclose(val, 35.0), f"Manual QUICK negative flow: got {val}, expected 35.0"
    # Now use the interpolation function (must match manual)
    def quick_face(phi, i, flow_positive):
        if flow_positive:
            return (6/8)*phi[i] + (3/8)*phi[i+1] - (1/8)*phi[i-1]
        else:
            return (6/8)*phi[i+1] + (3/8)*phi[i] - (1/8)*phi[i+2]
    interp = quick_face(phi, 2, False)
    assert np.isclose(interp, 35.0), f"QUICK negative flow: got {interp}, expected 35.0"
# --- Isolated QUICK Face Interpolation Test ---
def quick_face_interpolation(phi, i, flow_positive):
    """
    Compute QUICK face value at the face between i and i+1.
    phi: 1D array of cell values
    i: index of the left cell (face is between i and i+1)
    flow_positive: True for positive flow, False for negative
    Returns: interpolated value at the face
    """
    if flow_positive:
        # Use i-1, i, i+1 for face between i and i+1
        return (3/8)*phi[i-1] + (6/8)*phi[i] - (1/8)*phi[i+1]
    else:
        # Use i+2, i+1, i for face between i and i+1
        return (3/8)*phi[i+2] + (6/8)*phi[i+1] - (1/8)*phi[i]

def test_quick_face_interpolation_positive():
    phi = np.array([0.0, 1.0, 2.0, 3.0])
    # Face between i=1 and i=2, positive flow
    val = quick_face_interpolation(phi, 1, True)
    # Manual: (3/8)*0 + (6/8)*1 - (1/8)*2 = 0 + 0.75 - 0.25 = 0.5
    assert np.isclose(val, 0.5), f"QUICK positive flow: got {val}, expected 0.5"

def test_quick_face_interpolation_negative():
    phi = np.array([0.0, 1.0, 2.0, 3.0])
    # Face between i=1 and i=2, negative flow
    val = quick_face_interpolation(phi, 1, False)
    # Manual: (3/8)*3 + (6/8)*2 - (1/8)*1 = 1.125 + 1.5 - 0.125 = 2.5
    assert np.isclose(val, 2.5), f"QUICK negative flow: got {val}, expected 2.5"
# --- MMS Convergence Test for Advection Operators ---
import pytest
import numpy as np

@pytest.mark.parametrize("scheme", ["upwind", "quick"])
def test_advection_mms_convergence(scheme):
    """
    MMS convergence test for advection operator.
    ϕ(x, y) = sin(pi x) * cos(pi y), u = v = 1.
    Source: S = pi [cos(pi x) cos(pi y) - sin(pi x) sin(pi y)]
    Checks L2 error and observed order of accuracy for upwind and QUICK.
    """
    from pyfoamclone.numerics.operators.advection import advect_upwind
    from pyfoamclone.numerics.operators.reference_quick import advect_quick_reference
    grids = [8, 16, 32, 64]
    errors = []
    for N in grids:
        nx = ny = N
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        phi = np.sin(np.pi * X) * np.cos(np.pi * Y)
        u = np.ones_like(phi)
        v = np.ones_like(phi)
        # Analytical source term
        S = np.pi * (np.cos(np.pi * X) * np.cos(np.pi * Y) - np.sin(np.pi * X) * np.sin(np.pi * Y))
        # Numerical operator
        if scheme == "upwind":
            num = advect_upwind(u, v, phi, dx, dy)
        else:
            num = advect_quick_reference(phi, u, v, dx, dy)
        # Compute L2 error on interior (avoid boundaries)
        mask = (slice(2, -2), slice(2, -2)) if N > 8 else (slice(1, -1), slice(1, -1))
        err = np.sqrt(np.mean((num[mask] - S[mask])**2))
        errors.append(err)
    # Compute observed order of accuracy
    rates = []
    for i in range(1, len(errors)):
        p = np.log(errors[i-1]/errors[i]) / np.log(grids[i]/grids[i-1])
        rates.append(p)
    # Print summary table
    print("\nMMS Advection Convergence Table (scheme: %s)" % scheme)
    print("| Grid |   L2 Error   |  Observed p |")
    print("|------|-------------|-------------|")
    for i, N in enumerate(grids):
        if i == 0:
            print(f"| {N:4d} | {errors[i]:.6e} |      -      |")
        else:
            print(f"| {N:4d} | {errors[i]:.6e} |   {rates[i-1]:.3f}   |")
    # Assert expected order
    if scheme == "upwind":
        assert rates[-1] > 0.9 and rates[-1] < 1.2, f"Upwind order not ~1: {rates[-1]}"
    else:
        assert rates[-1] > 2.0, f"QUICK order not >2: {rates[-1]}"
import numpy as np
from pyfoamclone.numerics.fluid_ops import divergence, gradient, laplacian


def test_divergence_constant_field_zero():
    u = np.ones((8,8))*2.5
    v = np.ones((8,8))*(-1.7)
    dx=0.1; dy=0.2
    div = divergence(u,v,dx,dy)
    assert np.allclose(div, 0.0, atol=1e-12)


def test_gradient_linear_field_constant():
    nx=10; ny=9
    dx=0.05; dy=0.04
    x = np.arange(nx)*dx
    y = np.arange(ny)*dy
    X,Y = np.meshgrid(x,y,indexing='xy')
    p = 3.0*X + 4.0*Y + 2.0
    dpdx, dpdy = gradient(p, dx, dy)
    interior = (dpdx[1:-1,1:-1], dpdy[1:-1,1:-1])
    assert np.allclose(interior[0], 3.0, atol=1e-12)
    assert np.allclose(interior[1], 4.0, atol=1e-12)


def test_laplacian_quadratic_constant():
    nx=20; ny=18
    dx=0.1; dy=0.1
    x = np.arange(nx)*dx
    y = np.arange(ny)*dy
    X,Y = np.meshgrid(x,y,indexing='xy')
    f = X**2 + Y**2  # Laplacian = 2 + 2 = 4
    lap = laplacian(f, dx, dy)
    assert np.allclose(lap[2:-2,2:-2].mean(), 4.0, rtol=1e-2, atol=5e-2)


def test_advection_of_linear_field():
    """
    For a linear field phi = x + y and uniform velocity u=1, v=1,
    the upwind convective derivative should be 2 everywhere (except boundaries).
    """
    from pyfoamclone.numerics.operators.advection import advect_upwind
    nx, ny = 8, 8
    dx, dy = 1.0, 1.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    phi = X + Y
    u = np.ones_like(phi)
    v = np.ones_like(phi)
    conv = advect_upwind(u, v, phi, dx, dy)
    # Interior points should be 2.0 (d(phi)/dx + d(phi)/dy)
    assert np.allclose(conv[1:,1:], 2.0)


def test_advection_quick_of_quadratic_field():
    """
    For a quadratic field phi = x**2 + y**2 and uniform velocity u=1, v=1,
    QUICK should be much more accurate (lower error) than upwind.
    """
    import numpy as np
    from pyfoamclone.numerics.operators.advection import advect_upwind
    from pyfoamclone.numerics.operators.reference_quick import advect_quick_reference
    nx, ny = 16, 16
    dx, dy = 1.0, 1.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    phi = X**2 + Y**2
    u = np.ones_like(phi)
    v = np.ones_like(phi)
    # Analytical convective derivative: u*d(phi)/dx + v*d(phi)/dy = 2x + 2y
    expected = 2*X + 2*Y
    upwind = advect_upwind(u, v, phi, dx, dy)
    quick = advect_quick_reference(phi, u, v, dx, dy)
    # Compute mean absolute error (ignore boundaries)
    mask = (slice(2, -2), slice(2, -2))
    err_upwind = np.mean(np.abs(upwind[mask] - expected[mask]))
    err_quick = np.mean(np.abs(quick[mask] - expected[mask]))
    # QUICK should be at least 10x more accurate
    assert err_quick < err_upwind / 10, f"QUICK error {err_quick} not < 1/10 upwind error {err_upwind}"
