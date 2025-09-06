import pytest
import numpy as np
from pyflow_solver.solver import solve_lid_driven_cavity


def _run_validation_solver(N, Re, T):
    """Helper to run a longer, higher-res validation case safely."""
    u_res, v_res, p_res, residuals = None, None, None, None
    try:
        u_res, v_res, p_res, residuals = solve_lid_driven_cavity(
            N=N, Re=Re, dt=0.001, T=T, p_iterations=50
        )
    except Exception as e:
        pytest.fail(f"Solver crashed during validation run: {e}")

    if u_res is None or np.isnan(u_res).any():
        pytest.fail("Solver produced NaN in u-velocity during validation.")

    return u_res, v_res, p_res


@pytest.mark.slow
def test_validation_re100_comparison():
    """
    Validates the solver against benchmark data from Ghia et al. (1982) for Re=100.
    This test is marked as 'slow' and can be skipped with `pytest -m "not slow"`.
    """
    # Ghia et al. benchmark data for u-velocity along the vertical centerline (x=0.5)
    # for a 129x129 grid. We'll use a coarser grid and wider tolerances.
    ghia_y = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0])
    ghia_u = np.array([1.0, 0.65, 0.37, 0.17, 0.02, -0.10, -0.20, -0.32, -0.21, 0.0])

    N = 32  # A slightly finer grid for better comparison
    u, v, p = _run_validation_solver(N=N, Re=100.0, T=10.0)

    # Extract our solver's u-velocity profile at the vertical centerline
    center_x_idx = N // 2
    our_u_profile = u[:, center_x_idx]

    # Interpolate Ghia data to our grid points for comparison
    our_y = np.linspace(0, 1, N)
    ghia_u_interp = np.interp(our_y, ghia_y, ghia_u)

    # Compare the profiles using a tolerance
    # We use a fairly loose tolerance because our grid is much coarser.
    assert np.allclose(our_u_profile, ghia_u_interp, atol=0.1), \
        "U-velocity profile deviates significantly from Ghia benchmark data."


@pytest.mark.slow
def test_grid_independence_study():
    """
    A simple grid independence check. Doubling the grid resolution should
    result in a similar, but more resolved, flow field. The velocity at the
    center point should converge.
    """
    # Run with a coarse grid
    u_coarse, _, _ = _run_validation_solver(N=16, Re=100.0, T=5.0)
    center_u_coarse = u_coarse[8, 8]

    # Run with a finer grid
    u_fine, _, _ = _run_validation_solver(N=32, Re=100.0, T=5.0)
    center_u_fine = u_fine[16, 16]

    # The results should be reasonably close, with the finer grid being more accurate.
    # For this test, we just check they are within 20% of each other.
    assert np.isclose(center_u_coarse, center_u_fine, rtol=0.2), \
        f"Grid independence check failed. Coarse: {center_u_coarse}, Fine: {center_u_fine}"