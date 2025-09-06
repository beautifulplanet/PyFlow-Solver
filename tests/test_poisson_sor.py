import os, math
import numpy as np
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import pressure_rhs_unscaled, solve_pressure_poisson_unscaled, PROJECTION_BACKEND

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

def test_poisson_sor_converges_and_faster():
    os.environ['PROJECTION_ENABLE'] = '1'
    os.environ['PROJECTION_POISSON_BASE_TOL'] = '1e-5'
    # Baseline Jacobi
    os.environ['PROJECTION_LINSOLVER'] = 'jacobi'
    st_j = make_state(33)
    rhs_j = pressure_rhs_unscaled(st_j)
    it_j = solve_pressure_poisson_unscaled(rhs_j, st_j, tol=1e-5, max_iter=6000)
    # SOR
    os.environ['PROJECTION_LINSOLVER'] = 'sor'
    os.environ['PROJECTION_SOR_OMEGA'] = '1.7'
    st_s = make_state(33)
    rhs_s = pressure_rhs_unscaled(st_s)
    it_s = solve_pressure_poisson_unscaled(rhs_s, st_s, tol=1e-5, max_iter=6000)
    assert it_s < it_j, f"SOR not faster than Jacobi: sor={it_s} jacobi={it_j} backend={PROJECTION_BACKEND}"
    assert it_s < 0.9 * it_j, f"SOR speed gain insufficient: sor={it_s} jacobi={it_j}"
