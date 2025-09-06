import json
import numpy as np
import pytest

from pyfoamclone.core.ghost_fields import allocate_state, interior_view
from pyfoamclone.residuals.manager import ResidualManager
from pyfoamclone.solvers.solver import step
from pyfoamclone.configuration.loader import load_config
from pyfoamclone.configuration.config import SolverConfig


class PhysConfig(SolverConfig):
    # ensure physical solver tag for tests
    solver: str = "physical"


def test_dt_growth_limited():
    state = allocate_state(8, 8)
    cfg = PhysConfig(nx=8, ny=8, lx=1.0, ly=1.0, Re=1e6, cfl_growth=1.05)
    tracker = ResidualManager()
    # First step should clamp dt to prev_dt * growth (prev_dt=1e-2)
    _, _, diag = step(cfg, state, tracker, iteration=0)
    dt1 = state.meta['dt_prev']
    assert pytest.approx(dt1, rel=1e-12) == 1e-2 * cfg.cfl_growth
    # Second step: velocities still zero; dt should continue to grow but limited again
    _, _, diag2 = step(cfg, state, tracker, iteration=1)
    dt2 = state.meta['dt_prev']
    assert dt2 <= dt1 * cfg.cfl_growth + 1e-15


def test_matrix_caching_reuse():
    state = allocate_state(6, 6)
    cfg = PhysConfig(nx=6, ny=6)
    tracker = ResidualManager()
    step(cfg, state, tracker, 0)
    L_id_1 = id(state.meta['L_cache'])
    P_id_1 = id(state.meta['P_cache'])
    step(cfg, state, tracker, 1)
    assert id(state.meta['L_cache']) == L_id_1
    assert id(state.meta['P_cache']) == P_id_1


def test_boundary_conditions_enforced_after_step():
    state = allocate_state(10, 10)
    cfg = PhysConfig(nx=10, ny=10, lid_velocity=3.0)
    tracker = ResidualManager()
    # Corrupt top lid before step
    u = interior_view(state.fields['u'])
    u[-1,:] = -99.0
    step(cfg, state, tracker, 0)
    u_after = interior_view(state.fields['u'])
    v_after = interior_view(state.fields['v'])
    assert np.allclose(u_after[-1,:], 3.0)
    assert np.allclose(v_after[-1,:], 0.0)
    assert np.allclose(u_after[0,:], 0.0)


def test_linear_solver_iteration_cap():
    state = allocate_state(6, 6)
    cfg = PhysConfig(nx=6, ny=6, lin_maxiter=5, lin_tol=1e-12)
    tracker = ResidualManager()
    _, _, diag = step(cfg, state, tracker, 0)
    assert diag['lin_iter_u'] is None or diag['lin_iter_u'] <= cfg.lin_maxiter


def test_load_config_rejects_invalid_solver(tmp_path):
    cfg_path = tmp_path / 'bad.json'
    bad_cfg = {
        "schema_version": 1,
        "nx": 4,
        "ny": 4,
        "lx": 1.0,
        "ly": 1.0,
        "Re": 100.0,
        "solver": "invalid_solver_type",
        "max_iter": 5,
        "tol": 1e-6
    }
    cfg_path.write_text(json.dumps(bad_cfg))
    with pytest.raises(ValueError):
        load_config(str(cfg_path))


def test_manifest_version_fields(tmp_path):
    # Build minimal physical config file and run through run_case
    cfg_path = tmp_path / 'good.json'
    good_cfg = {
        "schema_version": 1,
        "nx": 4,
        "ny": 4,
        "lx": 1.0,
        "ly": 1.0,
        "Re": 50.0,
        "solver": "physical",
        "max_iter": 2,
        "tol": 1e-6
    }
    cfg_path.write_text(json.dumps(good_cfg))
    from pyfoamclone.scripts.run_case import run
    result = run(str(cfg_path))
    mf = result['manifest']
    assert 'version' in mf and 'git_sha' in mf and 'config_hash' in mf


def test_divergence_reduced_after_multiple_steps():
    state = allocate_state(12, 12)
    rng = np.random.default_rng(42)
    interior_view(state.fields['u'])[:] = rng.standard_normal((12,12))*0.05
    interior_view(state.fields['v'])[:] = rng.standard_normal((12,12))*0.05
    cfg = PhysConfig(nx=12, ny=12)
    tracker = ResidualManager()
    from pyfoamclone.numerics.fluid_ops import divergence
    initial = np.linalg.norm(divergence(state))
    for it in range(5):
        step(cfg, state, tracker, it)
    final = np.linalg.norm(divergence(state))
    assert final <= initial * 0.98  # at least 2% reduction overall


# --- NEW TEST: Does a single solver step reduce divergence by 100x? ---
def test_full_solver_step_reduces_divergence():
    """
    Create a simple 16x16 State with a shear flow (u = y, v = 0),
    compute initial divergence, run one solver step, and check that
    the divergence norm is reduced by at least 100x.
    """
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.residuals.manager import ResidualManager
    from pyfoamclone.solvers.solver import step
    from pyfoamclone.numerics.fluid_ops import divergence
    nx, ny = 16, 16
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    state = allocate_state(nx, ny)
    # Shear flow: u = y, v = 0
    u = interior_view(state.fields['u'])
    v = interior_view(state.fields['v'])
    for j in range(ny):
        u[:, j] = j * dy
    v[:, :] = 0.0
    # Compute initial divergence
    div0 = divergence(state)
    norm0 = np.linalg.norm(div0)
    # Run one solver step
    cfg = PhysConfig(nx=nx, ny=ny, lx=1.0, ly=1.0, Re=100.0)
    tracker = ResidualManager()
    step(cfg, state, tracker, 0)
    # Compute final divergence
    div1 = divergence(state)
    norm1 = np.linalg.norm(div1)
    print(f"Initial divergence norm: {norm0}")
    print(f"Final divergence norm: {norm1}")
    assert norm1 < norm0 * 1e-2, f"Divergence not reduced enough: {norm1} vs {norm0}"
