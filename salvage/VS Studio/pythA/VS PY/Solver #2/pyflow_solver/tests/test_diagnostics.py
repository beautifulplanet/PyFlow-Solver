import numpy as np
import pytest
import matplotlib.pyplot as plt

from pyflow.solver import solve_lid_driven_cavity
from pyflow.logging import LiveLogger
from pyflow.grid import Grid


@pytest.mark.timeout(10)
def test_boundary_conditions_enforced(capsys):
    N = 16
    Re = 100.0
    T = 0.05
    dt = 0.005
    dx = dy = 1.0 / (N - 1)
    logger = LiveLogger(N, Re, dt, T)
    with capsys.disabled():
        u, v, p = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, logger=logger)
    
    # Lid: top row, except corners, should be 1.0
    assert np.allclose(u[-1,1:-1], 1.0, atol=1e-8)
    # Bottom, left, right: should be 0
    assert np.allclose(u[0,:], 0.0, atol=1e-8)
    assert np.allclose(u[:,0], 0.0, atol=1e-8)
    assert np.allclose(u[:,-1], 0.0, atol=1e-8)
    # Corners: top-left and top-right should be 0
    assert u[-1,0] == 0.0
    assert u[-1,-1] == 0.0
    # v boundaries
    assert np.allclose(v[0,:], 0.0, atol=1e-8)
    assert np.allclose(v[-1,:], 0.0, atol=1e-8)
    assert np.allclose(v[:,0], 0.0, atol=1e-8)
    assert np.allclose(v[:,-1], 0.0, atol=1e-8)


@pytest.mark.timeout(60)
def test_flow_develops(capsys):
    N = 16
    Re = 100.0
    T = 0.5  # Run for a longer time to allow flow to develop
    dt = 0.005
    dx = dy = 1.0 / (N - 1)
    logger = LiveLogger(N, Re, dt, T)
    with capsys.disabled():
        u, v, p = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, p_iterations=5000, logger=logger)
    
    # Check that the interior is not all zero (flow develops)
    assert np.any(np.abs(u[1:-1,1:-1]) > 1e-6)
    assert np.any(np.abs(v[1:-1,1:-1]) > 1e-6)


@pytest.mark.timeout(10)
def test_pressure_field_nontrivial(capsys):
    N = 16
    Re = 100.0
    T = 0.05
    dt = 0.005
    dx = dy = 1.0 / (N - 1)
    logger = LiveLogger(N, Re, dt, T)
    with capsys.disabled():
        u, v, p = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, logger=logger)
    
    # Pressure field should not be constant
    assert np.std(p) > 1e-6

@pytest.mark.timeout(120)
def test_lid_driven_cavity_center_velocity(capsys):
    """
    Test the centerline velocity for a classic lid-driven cavity at Re=100.
    Reference: Ghia et al. (1982) benchmark data.
    """
    NPOINTS, T = 16, 7.0
    L = 1.0
    Re = 100.0
    dt = 0.001
    
    grid = Grid(NPOINTS, L)
    logger = LiveLogger(NPOINTS, Re, dt, T, log_interval=100)

    with capsys.disabled():
        u, v, p = solve_lid_driven_cavity(grid.NPOINTS, grid.dx, grid.dy, Re, dt, T, p_iterations=5000, logger=logger)

    # Check for NaN/Inf
    assert not np.isnan(u).any() and not np.isinf(u).any(), "u field contains NaN or Inf"
    assert not np.isnan(v).any() and not np.isinf(v).any(), "v field contains NaN or Inf"

    center_idx = NPOINTS // 2
    v_centerline = v[center_idx, :]
    
    # Ghia et al. (1982) value for v at (x=0.5, y=0.5) for Re=100 is about 0.00333
    # We use a loose tolerance for this coarse grid and short simulation time.
    assert np.isclose(v_centerline[center_idx], 0.00333, atol=0.02), \
        f"Centerline v-velocity {v_centerline[center_idx]} deviates from benchmark."
