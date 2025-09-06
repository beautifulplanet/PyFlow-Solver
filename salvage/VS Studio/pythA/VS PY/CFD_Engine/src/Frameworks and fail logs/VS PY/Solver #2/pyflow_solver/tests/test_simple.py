import numpy as np
import pytest
from pyflow.solver import solve_lid_driven_cavity

def test_single_step_execution():
    """
    Tests if the solver can execute a single timestep without crashing.
    """
    N = 16
    Re = 10.0
    T = 0.001
    dt = 0.001
    dx = dy = 1.0 / (N - 1)
    
    try:
        u, v, p = solve_lid_driven_cavity(N, dx, dy, Re, dt, T, p_iterations=10)
        # Check if outputs are valid numpy arrays with the correct shape
        assert isinstance(u, np.ndarray)
        assert u.shape == (N, N)
        assert not np.isnan(u).any()
        assert not np.isinf(u).any()

        assert isinstance(v, np.ndarray)
        assert v.shape == (N, N)
        assert not np.isnan(v).any()
        assert not np.isinf(v).any()

        assert isinstance(p, np.ndarray)
        assert p.shape == (N, N)
        assert not np.isnan(p).any()
        assert not np.isinf(p).any()

    except Exception as e:
        pytest.fail(f"Solver crashed on a single step with a simple configuration: {e}")

