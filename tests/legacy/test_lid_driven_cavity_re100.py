"""Lid-driven cavity qualitative/quantitative smoke tests (Re~100).

Pyright/Pylance note: A prior transient cache produced a spurious
"Variable not allowed in type expression" warning pointing at line 20.
No invalid type expressions exist; suppress at file level if it recurs.
"""

# pyright: reportInvalidTypeForm=false

import os, math
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step

# Reference sign expectations (very crude smoke test) for Re=100 (nu chosen)
# We'll just ensure qualitative velocity profile features appear after some steps.


def make_state(n=33):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01, rho=1.0)
    st.require_field('u', (n,n))
    st.require_field('v', (n,n))
    st.require_field('p', (n,n))
    return st


def cavity_bc(state: SolverState):
    u = state.fields['u']; v = state.fields['v']
    # No-slip walls
    u[0,:] = 0.0; u[-1,:] = 0.0; u[:,0] = 0.0; u[:,-1] = 1.0  # moving lid on top (y=1)
    v[0,:] = 0.0; v[-1,:] = 0.0; v[:,0] = 0.0; v[:,-1] = 0.0


def divergence(u,v,dx,dy):
    return ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))

@pytest.mark.slow
def test_lid_driven_cavity_basic_structure():
    os.environ['PROJECTION_ENABLE'] = '1'
    os.environ.pop('PROJECTION_LINSOLVER', None)  # default jacobi for determinism
    st = make_state()
    n = st.mesh.nx
    max_steps = 400
    dt = 1e-3  # larger dt to accelerate shear diffusion while remaining stable for this coarse grid
    achieved_neg = False
    for step in range(max_steps):
        projection_step(st, dt=dt, use_advection=True, adaptive_dt=False, boundary_fn=cavity_bc)
        if step % 20 == 0:  # periodic early check
            mid_i = n//2
            u_col_tmp = st.fields['u'][mid_i,:]
            if np.min(u_col_tmp[2:-2]) < -5e-4:
                achieved_neg = True
                break
    # Sample vertical centerline u profile (x ~ 0.5)
    mid_i = n//2
    u_col = st.fields['u'][mid_i,:]
    # Expect primary vortex: near bottom u ~ 0, somewhere mid-height negative, near top positive (due to lid shear)
    assert u_col[1] <= 0.05  # near zero at bottom
    min_mid = float(np.min(u_col[2:-2]))
    assert min_mid < -5e-4, f"Insufficient mid negative recirculation (min={min_mid})"
    assert u_col[-2] > 0.02, f"Lid shear not imparted (u near lid={u_col[-2]})"
    # Divergence remains small
    dx=st.mesh.dx(); dy=st.mesh.dy()
    div = divergence(st.fields['u'], st.fields['v'], dx, dy)
    assert float(np.max(np.abs(div))) < 5.0  # loose bound

@pytest.mark.slow
def test_lid_driven_cavity_centerline_profiles():
    """Quantitative coarse Re=100 centerline profile check (fast variant).

    Early-exits once characteristic recirculation signatures stabilize to keep
    runtime low while still asserting physically meaningful structure.
    """
    # LEGACY (will be moved to tests/legacy)
    os.environ['PROJECTION_ENABLE'] = '1'
    st = make_state()
    n = st.mesh.nx
    dt = 1e-3
    max_steps = 350  # reduced from 500 for speed
    target_min_reached = False
    last_min = None
    stable_count = 0
    for step in range(max_steps):
        projection_step(st, dt=dt, use_advection=True, adaptive_dt=False, boundary_fn=cavity_bc)
        if step % 25 == 0 and step > 0:
            mid_i = n//2
            u_col_tmp = st.fields['u'][mid_i,:]
            curr_min = float(np.min(u_col_tmp[2:-2]))
            # Check if negative recirculation established
            if curr_min < -0.02:
                target_min_reached = True
                if last_min is not None and abs(curr_min - last_min) < 0.002:
                    stable_count += 1
                else:
                    stable_count = 0
                last_min = curr_min
                if stable_count >= 2:  # three plateau measurements (including current)
                    break
    # Final sampling
    mid_i = n//2; mid_j = n//2
    u_col = st.fields['u'][mid_i,:]
    v_row = st.fields['v'][:,mid_j]
    min_u = float(np.min(u_col[2:-2]))
    assert target_min_reached, "Recirculation minimum not reached within step budget"
    assert -0.18 < min_u < -0.02, f"u centerline min out of expected coarse range: {min_u}"
    u_lid = float(u_col[-2])
    assert 0.5 < u_lid < 1.25, f"u near lid unrealistic: {u_lid}"
    min_v = float(np.min(v_row[2:-2]))
    assert min_v < -0.008, f"v centerline lacks downward flow signature: {min_v}"
    min_index = int(np.argmin(v_row[2:-2])) + 2
    assert 3 < min_index < n-4, "v minimum at boundary indicates incorrect recirculation structure"
    dx=st.mesh.dx(); dy=st.mesh.dy()
    div = divergence(st.fields['u'], st.fields['v'], dx, dy)
    assert float(np.max(np.abs(div))) < 10.0
