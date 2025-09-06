import pytest
from pyfoamclone.linear_solvers.multigrid import multigrid_solver

def test_multigrid_solver_placeholder():
    with pytest.raises(NotImplementedError):
        multigrid_solver(None, None)
