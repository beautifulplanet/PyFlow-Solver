import os, math
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step


def divergence(u,v,dx,dy):
    return ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))


def make_state(n, field_fn):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01)
    u = st.require_field('u', (n,n))
    v = st.require_field('v', (n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    uf, vf = field_fn(X,Y)
    u[:,:] = uf
    v[:,:] = vf
    st.require_field('p', (n,n))
    return st

@pytest.fixture(autouse=True)
def enable_projection(monkeypatch):
    monkeypatch.setenv('PROJECTION_ENABLE','1')


def test_projection_random_field_reduces_divergence():
    rng = np.random.default_rng(42)
    n = 31
    def rand_field(X,Y):
        return rng.standard_normal(X.shape), rng.standard_normal(X.shape)
    st = make_state(n, rand_field)
    dx = st.mesh.dx(); dy = st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf0 = float(np.max(np.abs(div0)))
    projection_step(st, dt=5e-4, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    assert linf1 < linf0, f"Random field divergence not reduced linf0={linf0} linf1={linf1}"


def test_projection_idempotent():
    n = 33
    def field(X,Y):
        return np.sin(math.pi*X), np.cos(math.pi*Y)
    st = make_state(n, field)
    dx = st.mesh.dx(); dy = st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    projection_step(st, dt=2e-4, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    # apply again
    projection_step(st, dt=2e-4, use_advection=False)
    div2 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    linf2 = float(np.max(np.abs(div2)))
    # Second pass should not worsen divergence materially
    assert linf2 <= linf1 * 1.05 + 1e-10, f"Projection not idempotent enough: linf1={linf1} linf2={linf2}"


def test_projection_preserves_div_free_field():
    n = 33
    def div_free(X,Y):
        # u = sin(pi x) cos(pi y); v = -cos(pi x) sin(pi y) -> divergence=0
        return np.sin(math.pi*X)*np.cos(math.pi*Y), -np.cos(math.pi*X)*np.sin(math.pi*Y)
    st = make_state(n, div_free)
    dx = st.mesh.dx(); dy = st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf0 = float(np.max(np.abs(div0)))
    projection_step(st, dt=2e-4, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    assert linf0 < 1e-10 and linf1 < 5e-3, f"Divergence-free field degraded linf0={linf0} linf1={linf1}"


def test_projection_small_grid():
    n = 5
    def field(X,Y):
        return np.sin(2*math.pi*X), np.sin(2*math.pi*Y)
    st = make_state(n, field)
    dx = st.mesh.dx(); dy = st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf0 = float(np.max(np.abs(div0)))
    projection_step(st, dt=1e-3, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    assert linf1 < linf0, f"Small grid divergence not reduced linf0={linf0} linf1={linf1}"
