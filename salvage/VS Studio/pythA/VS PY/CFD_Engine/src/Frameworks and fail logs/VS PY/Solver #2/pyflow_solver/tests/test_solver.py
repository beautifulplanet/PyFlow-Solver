import sys
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyflow.solver import solve_lid_driven_cavity
from pyflow.logging import LiveLogger
from pyflow.grid import Grid

# Try to import the hybrid solver
try:
    from pyflow.hybrid_solver import solve_lid_driven_cavity as solve_hybrid
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False


@pytest.mark.timeout(10)
def test_boundary_conditions_enforced(capsys):
    N = 16
    Re = 100.0
    T = 0.05
    dt = 0.005
    dx = dy = 1.0 / (N - 1)
    logger = LiveLogger(N, Re, dt, T)
    with capsys.disabled():
        u, v, p, residuals = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, logger=logger)
    
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


@pytest.mark.timeout(10)
def test_flow_develops(capsys):
    N = 16
    Re = 100.0
    T = 0.05
    dt = 0.005
    dx = dy = 1.0 / (N - 1)
    logger = LiveLogger(N, Re, dt, T)
    with capsys.disabled():
        u, v, p, residuals = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, logger=logger)
    
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
        u, v, p, residuals = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, logger=logger)
    
    # Pressure field should not be constant
    assert np.std(p) > 1e-6

@pytest.mark.timeout(20)
def test_lid_driven_cavity_center_velocity(capsys):
    """
    Test the centerline velocity for a classic lid-driven cavity at Re=100.
    Reference: Ghia et al. (1982) benchmark data.
    """
    NPOINTS, T = 16, 0.1
    L = 1.0
    Re = 100.0
    dt = 0.005
    
    grid = Grid(NPOINTS, L)
    logger = LiveLogger(NPOINTS, Re, dt, T)

    with capsys.disabled():
        u, v, p, residuals = solve_lid_driven_cavity(grid.NPOINTS, grid.dx, grid.dy, Re, dt, T, logger=logger)

    # Check for NaN/Inf
    assert not np.isnan(u).any() and not np.isinf(u).any(), "u field contains NaN or Inf"
    assert not np.isnan(v).any() and not np.isinf(v).any(), "v field contains NaN or Inf"

    center_idx = NPOINTS // 2
    u_centerline = u[:, center_idx]
    
    # Ghia et al. (1982) value for u at (x=0.5, y=0.5) for Re=100 is about 0.062
    # We use a loose tolerance for this coarse grid and short simulation time.
    assert np.isclose(u_centerline[center_idx], 0.062, atol=0.02), \
        f"Centerline u-velocity {u_centerline[center_idx]} deviates from benchmark."

@pytest.mark.timeout(30)
@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid solver not available")
def test_hybrid_solver_correctness(capsys):
    """
    Test that the hybrid solver produces results consistent with the pure Python solver.
    This ensures the C++ components behave correctly.
    """
    # Use smaller values for a quick test
    NPOINTS, T = 17, 0.15
    L = 1.0
    Re = 100.0
    dt = 0.002
    
    grid = Grid(NPOINTS, L)
    logger = LiveLogger(NPOINTS, Re, dt, T, log_interval=100)
    
    # Run both solvers with identical parameters
    with capsys.disabled():
        print("Running Python solver...")
        u_py, v_py, p_py, _ = solve_lid_driven_cavity(
            grid.NPOINTS, grid.dx, grid.dy, Re, dt, T,
            p_iterations=100,  # Use fewer iterations for speed
            logger=logger
        )
        
        print("Running hybrid solver...")
        u_hybrid, v_hybrid, p_hybrid, _ = solve_hybrid(
            grid.NPOINTS, grid.dx, grid.dy, Re, dt, T,
            p_iterations=100,  # Use fewer iterations for speed
            logger=logger
        )
    
    # Compare results - should be identical or very close
    # We use slightly relaxed tolerances to account for floating point differences
    # between Python and C++ implementations
    u_diff = np.max(np.abs(u_py - u_hybrid))
    v_diff = np.max(np.abs(v_py - v_hybrid))
    p_diff = np.max(np.abs(p_py - p_hybrid))
    
    # These tolerances are relatively generous for initial testing
    assert u_diff < 0.05, f"u-velocity maximum difference too large: {u_diff}"
    assert v_diff < 0.05, f"v-velocity maximum difference too large: {v_diff}"
    assert p_diff < 0.1, f"pressure maximum difference too large: {p_diff}"
    
    # Check specific points for closer agreement
    center_idx = NPOINTS // 2
    u_centerline_py = u_py[:, center_idx]
    u_centerline_hybrid = u_hybrid[:, center_idx]
    
    # Center point should be very close
    assert np.isclose(u_centerline_py[center_idx], u_centerline_hybrid[center_idx], atol=0.01), \
        f"Center u-velocity differs: Python={u_centerline_py[center_idx]}, Hybrid={u_centerline_hybrid[center_idx]}"
