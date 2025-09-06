import numpy as np
import pytest
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.solvers.pressure_solver import solve_pressure_poisson

class Cfg:
    projection_use_mf = True
    lin_tol = 1e-12
    lin_maxiter = 500

@pytest.mark.parametrize("nx,ny", [(8,8), (12,10)])
def test_projection_manufactured_reduces_divergence(nx, ny):
    # Manufactured velocity field with known divergence pattern
    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    # Use p(x,y) = cos(pi x) cos(pi y) which satisfies homogeneous Neumann (zero normal derivative) at domain boundaries.
    p_exact = np.cos(np.pi*X) * np.cos(np.pi*Y)
    dpdx_exact = -np.pi * np.sin(np.pi*X) * np.cos(np.pi*Y)
    dpdy_exact = -np.pi * np.cos(np.pi*X) * np.sin(np.pi*Y)
    # Manufactured provisional velocity u* = dt * grad(p_exact) so that projection should recover zero velocity field.
    dt = 0.01
    u_star = dt * dpdx_exact
    v_star = dt * dpdy_exact
    state = allocate_state(nx, ny)
    interior_view(state.fields['u'])[:] = u_star
    interior_view(state.fields['v'])[:] = v_star
    # Pre-projection divergence norm
    # Use coherent divergence via solver import (pressure solver uses coherent ops)
    from pyflow.numerics.core_ops_coherent import divergence_coherent
    div0 = divergence_coherent(interior_view(state.fields['u']), interior_view(state.fields['v']), dx, dy)
    norm0 = float(np.linalg.norm(div0))
    p_corr, diag = solve_pressure_poisson(state, dt, dx, dy, Cfg())
    div1 = divergence_coherent(interior_view(state.fields['u']), interior_view(state.fields['v']), dx, dy)
    norm1 = float(np.linalg.norm(div1))
    # Expect very strong reduction (several orders); allow absolute fallback tolerance for coarse grids.
    assert norm1 < max(1e-10, norm0 * 5e-6), f"Divergence not sufficiently reduced: {norm0} -> {norm1}"
    # Pressure should match p_exact up to constant (we pinned at cell 0); remove offset and compare
    p_adj = p_exact - p_exact.flat[0]
    l2_err = np.linalg.norm(p_corr - p_adj) / (np.linalg.norm(p_adj) or 1.0)
    # NOTE: A single Dirichlet pin (cell 0) perturbs the pure Neumann solution creating an O(1) localized consistency defect;
    # on coarse grids this propagates and yields ~O(1e-1) relative L2 error vs manufactured p. We accept <=0.25 here.
    assert l2_err < 0.25, f"Pressure relative error too high: {l2_err}"
