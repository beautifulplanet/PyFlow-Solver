import os
import numpy as np
from cfd_solver.pyflow.core import Mesh, SolverState, projection_step

def make_state(n=33):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01)
    u = st.require_field('u',(n,n)); v = st.require_field('v',(n,n)); st.require_field('p',(n,n))
    # Random initial field
    u[:] = np.random.randn(*u.shape)
    v[:] = np.random.randn(*v.shape)
    return st

def test_projection_residual_within_factor():  # LEGACY (will be moved to tests/legacy)
    os.environ['PROJECTION_ENABLE']='1'
    os.environ['PROJECTION_LINSOLVER']='jacobi'
    os.environ['PROJECTION_POISSON_BASE_TOL']='1e-6'
    st = make_state()
    stats = projection_step(st, dt=2e-4, use_advection=False, adaptive_dt=False)
    tol = float(stats.notes['poisson_tol'])
    resid = stats.notes['poisson_resid_linf']
    assert resid <= 5.0 * tol, f"Residual {resid} exceeds 5x tol {tol}"  # generous guard

def test_projection_residual_assertion_flag():
    os.environ['PROJECTION_ENABLE']='1'
    os.environ['PROJECTION_ASSERT_RESID']='1'
    os.environ['PROJECTION_RESID_FACTOR']='3.0'
    st = make_state()
    # Should not raise given generous factor
    projection_step(st, dt=2e-4, use_advection=False, adaptive_dt=False)
    os.environ.pop('PROJECTION_ASSERT_RESID', None)
    os.environ.pop('PROJECTION_RESID_FACTOR', None)
