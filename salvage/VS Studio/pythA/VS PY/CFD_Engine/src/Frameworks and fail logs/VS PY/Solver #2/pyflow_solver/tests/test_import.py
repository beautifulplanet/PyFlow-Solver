import pytest

def test_import_solver():
    """
    Tests if the solver module can be imported without crashing the interpreter.
    """
    try:
        from pyflow.solver import solve_lid_driven_cavity
    except Exception as e:
        pytest.fail(f"Failed to import solve_lid_driven_cavity: {e}")
    assert True
