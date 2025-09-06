import os, math
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import pressure_rhs_unscaled, solve_pressure_poisson_unscaled


def make_state(n=33):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01, rho=1.0)
    u = st.require_field('u',(n,n)); v = st.require_field('v',(n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u[:,:] = np.sin(math.pi*X)
    v[:,:] = np.cos(math.pi*Y)
    st.require_field('p',(n,n))
    return st

@pytest.mark.slow
def test_multigrid_converges_and_faster():
    os.environ['PROJECTION_ENABLE'] = '1'
    # Baseline Jacobi
    os.environ['PROJECTION_LINSOLVER'] = 'jacobi'
    st_j = make_state(65)
    rhs_j = pressure_rhs_unscaled(st_j)
    it_j = solve_pressure_poisson_unscaled(rhs_j, st_j, tol=1e-5, max_iter=8000)
    # Multigrid
    os.environ['PROJECTION_LINSOLVER'] = 'mg'
    st_m = make_state(65)
    rhs_m = pressure_rhs_unscaled(st_m)
    it_m = solve_pressure_poisson_unscaled(rhs_m, st_m, tol=1e-5, max_iter=50)  # cycles
    assert it_m < it_j * 0.6, f"Multigrid not sufficiently faster: mg={it_m} jacobi={it_j}"
