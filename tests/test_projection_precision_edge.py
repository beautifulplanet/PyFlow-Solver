import os
import numpy as np
from cfd_solver.pyflow.core import Mesh, SolverState, projection_step, cfl_dt

def make_state(n=17, nu=0.01):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=nu)
    st.require_field('u',(n,n)); st.require_field('v',(n,n)); st.require_field('p',(n,n))
    return st

def test_dt_clamp_and_near_zero_velocity():
    os.environ['PROJECTION_ENABLE']='1'
    st = make_state(n=17, nu=0.01)
    st.fields['u'][:] = 1e-12
    st.fields['v'][:] = -1e-12
    dx = st.mesh.dx(); dy = st.mesh.dy()
    # Direct CFL dt should be diffusion-limited
    diff_limit = 0.25 * min(dx,dy)**2 / st.nu
    dt_cfl = cfl_dt(st.fields['u'], st.fields['v'], dx, dy, st.nu, cfl=0.5)
    assert abs(dt_cfl - diff_limit) / diff_limit < 1e-12
    stats = projection_step(st, dt=None, use_advection=False)  # adaptive dt
    assert stats.dt >= 1e-12, "dt clamp failed (too small)"

def test_negative_dt_input_clamped():
    os.environ['PROJECTION_ENABLE']='1'
    st = make_state()
    st.fields['u'][:] = np.random.randn(*st.fields['u'].shape)
    st.fields['v'][:] = np.random.randn(*st.fields['v'].shape)
    stats = projection_step(st, dt=-1e-2, use_advection=False, adaptive_dt=False)
    assert stats.dt > 0, "Negative dt not clamped to positive"