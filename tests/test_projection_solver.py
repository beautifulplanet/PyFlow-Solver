import os, math
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step
from cfd_solver.pyflow.core import cfl_dt

def make_state(n=17):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01)
    # Initialize a synthetic non-divergence-free velocity field
    u = st.require_field('u', (n,n))
    v = st.require_field('v', (n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u[:,:] = np.sin(math.pi*X)
    v[:,:] = np.cos(math.pi*Y)
    st.require_field('p', (n,n))
    return st

def divergence(u,v,dx,dy):
    return ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))


def test_projection_flag_block():
    st = make_state()
    if 'PROJECTION_ENABLE' in os.environ:
        del os.environ['PROJECTION_ENABLE']
    with pytest.raises(RuntimeError):
        projection_step(st, dt=1e-3)

@pytest.mark.slow
def test_projection_reduces_divergence(monkeypatch):
    monkeypatch.setenv('PROJECTION_ENABLE','1')
    st = make_state(n=33)
    dx = st.mesh.dx(); dy = st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf0 = float(np.max(np.abs(div0)))
    # Use smaller stable dt to avoid overflow during predictor
    stats = projection_step(st, dt=2e-4, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    # Require a meaningful reduction factor (guards against trivial or noise-level changes)
    assert linf1 < 0.3 * linf0, f"Divergence reduction insufficient: before {linf0:.3e} after {linf1:.3e}"
    # Kinetic energy should not increase when using pure projection without advection
    u = st.fields['u']; v = st.fields['v']
    ke0 = 0.5 * np.mean((u**2 + v**2))  # after projection step (no pre snapshot saved)
    # Re-run one more projection step with dt=0 (no predictor) to measure neutrality; dt tiny to trigger only correction
    div_before = divergence(u, v, dx, dy)
    linf_before = float(np.max(np.abs(div_before)))
    projection_step(st, dt=1e-8, use_advection=False)  # essentially pure pressure correction
    div_after = divergence(u, v, dx, dy)
    ke1 = 0.5 * np.mean((u**2 + v**2))
    # Divergence should not worsen and energy should not rise by more than a tiny tolerance
    assert float(np.max(np.abs(div_after))) <= linf_before * 1.05, "Projection step worsened divergence"
    assert ke1 <= ke0 * 1.01, f"Kinetic energy unexpectedly increased: before {ke0} after {ke1}"

def test_adaptive_dt_reacts_to_velocity(monkeypatch):
    monkeypatch.setenv('PROJECTION_ENABLE','1')
    st = make_state(n=33)
    # Small velocity amplitude -> larger dt
    st.fields['u'] *= 0.05
    st.fields['v'] *= 0.05
    # Use direct CFL calculator to isolate dt logic from field evolution side-effects
    dx = st.mesh.dx(); dy = st.mesh.dy()
    dt1 = cfl_dt(st.fields['u'], st.fields['v'], dx, dy, st.nu, cfl=0.5)
    # Increase velocity magnitude -> expect smaller dt chosen
    st.fields['u'] *= 20.0  # net 1.0x original amplitude (0.05 * 20)
    st.fields['v'] *= 20.0
    dt2 = cfl_dt(st.fields['u'], st.fields['v'], dx, dy, st.nu, cfl=0.5)
    assert dt2 < dt1, f"Adaptive dt not reduced with higher velocity: dt1={dt1} dt2={dt2}"
