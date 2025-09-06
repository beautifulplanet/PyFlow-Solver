import os, math, statistics, random
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step

# Shared helpers

def make_state(n, u_expr, v_expr):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01)
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u = st.require_field('u', (n,n))
    v = st.require_field('v', (n,n))
    u[:,:] = u_expr(X,Y)
    v[:,:] = v_expr(X,Y)
    st.require_field('p', (n,n))
    return st

def divergence(u,v,dx,dy):
    return ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))

@pytest.fixture(autouse=True)
def enable_projection(monkeypatch):
    monkeypatch.setenv('PROJECTION_ENABLE','1')

# 1. Energy monotonicity (projection should not add significant kinetic energy)

def kinetic_energy(u,v):
    return 0.5 * float(np.sum(u*u + v*v))

def test_projection_energy_not_inflated():
    n=33
    st = make_state(n, lambda X,Y: np.sin(math.pi*X), lambda X,Y: np.cos(math.pi*Y))
    ke0 = kinetic_energy(st.fields['u'], st.fields['v'])
    projection_step(st, dt=2e-4, use_advection=False)
    ke1 = kinetic_energy(st.fields['u'], st.fields['v'])
    # Allow tiny numerical increase (0.5%)
    assert ke1 <= ke0 * 1.005, f"Kinetic energy increased too much ke0={ke0} ke1={ke1}"

# 2. Performance budget (iterations reasonable on baseline grid)

def test_projection_iteration_budget():
    n=33
    st = make_state(n, lambda X,Y: np.sin(math.pi*X), lambda X,Y: np.cos(math.pi*Y))
    stats = projection_step(st, dt=2e-4, use_advection=False)
    assert stats.iters <= 6000, f"Iteration budget exceeded: {stats.iters}"

# 3. Random dt fuzz (divergence reduction invariant)

def random_field_state(n, seed):
    rng = np.random.default_rng(seed)
    return make_state(n, lambda X,Y: rng.standard_normal(X.shape), lambda X,Y: rng.standard_normal(X.shape))

@pytest.mark.slow
def test_projection_random_dt_fuzz():
    n=33
    seeds = [101, 202, 303]
    dts = [5e-5, 1e-4, 3e-4, 7e-4]
    for s in seeds:
        for dt in dts:
            st = random_field_state(n, s)
            dx=st.mesh.dx(); dy=st.mesh.dy()
            div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
            linf0 = float(np.max(np.abs(div0)))
            projection_step(st, dt=dt, use_advection=False)
            div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
            linf1 = float(np.max(np.abs(div1)))
            assert linf1 < linf0, f"Fuzz divergence not reduced seed={s} dt={dt} before={linf0} after={linf1}"

# 4. Distribution regression guard

def test_projection_reduction_distribution():
    n=25
    ratios=[]
    for seed in range(10):
        st = random_field_state(n, seed)
        dx=st.mesh.dx(); dy=st.mesh.dy()
        div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
        linf0 = float(np.max(np.abs(div0)))
        projection_step(st, dt=2e-4, use_advection=False)
        div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
        linf1 = float(np.max(np.abs(div1)))
        ratios.append(linf1/linf0 if linf0>0 else 0.0)
    median = statistics.median(ratios)
    worst = max(ratios)
    assert median < 0.85, f"Median reduction ratio too high median={median} ratios={ratios}"
    assert worst < 0.97, f"Worst reduction ratio regression worst={worst} ratios={ratios}"

# 5. Larger grid stress (still reduces divergence)

@pytest.mark.slow
def test_projection_large_grid_divergence():
    n=65
    st = make_state(n, lambda X,Y: np.sin(math.pi*X), lambda X,Y: np.cos(math.pi*Y))
    dx=st.mesh.dx(); dy=st.mesh.dy()
    div0 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf0 = float(np.max(np.abs(div0)))
    stats = projection_step(st, dt=2e-4, use_advection=False)
    div1 = divergence(st.fields['u'], st.fields['v'], dx, dy)
    linf1 = float(np.max(np.abs(div1)))
    assert linf1 < linf0, f"Large grid divergence not reduced before={linf0} after={linf1}"
    # Informational: iterations may hit cap; accept but keep sanity upper bound
    assert stats.iters <= 8000, f"Iterations exceeded cap unexpectedly ({stats.iters})"
