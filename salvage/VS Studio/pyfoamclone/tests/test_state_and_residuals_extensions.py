import os
import json
import numpy as np
import pytest

from pyfoamclone.core.ghost_fields import allocate_state, interior_view, State
from pyfoamclone.residuals.manager import ResidualManager
from pyfoamclone.solvers.solver import step
from pyfoamclone.scripts.run_case import run as run_case


class DummyConfig:
    # minimal attributes consumed by solver.step and run_loop
    cfl_target = 0.5
    cfl_growth = 1.1
    tol = 1e-12
    max_iter = 3
    solver = 'physical'
    nx = 8
    ny = 8
    lx = 1.0
    ly = 1.0
    Re = 50.0


def test_state_metadata_persistence():
    state = allocate_state(6, 5)
    tracker = ResidualManager()
    cfg = DummyConfig()
    state, residuals, diag = step(cfg, state, tracker, iteration=0)
    assert 'dt_prev' in state.meta and state.meta['dt_prev'] > 0


def test_residual_manager_slope_plateau_detect_equivalence():
    rm = ResidualManager()
    name = 'r'
    # Construct window of 60: first 30 decreasing, last 30 nearly flat with slight upward drift
    dec = np.linspace(1.0, 0.3, 30)
    flat = 0.30 + 0.0002 * np.arange(30)  # gentle upward slope
    series = np.concatenate([dec, flat])
    for v in series:
        rm.add(name, float(v))
    # choose window = 40 (requires >=40 samples)
    assert rm.plateau(name, window=40) is True
    assert rm.plateau_detect(name, window=40, threshold=-0.01) is True


def test_residual_drop_orders_diagnostic():
    state = allocate_state(5, 5)
    tracker = ResidualManager()
    cfg = DummyConfig()
    # First step
    state, residuals, diag1 = step(cfg, state, tracker, iteration=0)
    # Artificially perturb u to create a larger second residual drop
    interior_view(state.fields['u'])[:] *= 0.5
    state, residuals, diag2 = step(cfg, state, tracker, iteration=1)
    drop_orders = diag2['residual_drop_Ru_orders']
    assert drop_orders >= 0.0
    assert isinstance(drop_orders, float)


def test_continuity_residual_small_after_steps():
    state = allocate_state(6, 6)
    tracker = ResidualManager()
    cfg = DummyConfig()
    for it in range(3):
        state, residuals, diag = step(cfg, state, tracker, iteration=it)
    # Continuity placeholder should remain very small (near zero)
    assert residuals['continuity'] < 1e-8


def test_allocate_state_custom_fields():
    state = allocate_state(4, 4, fields=("phi", "T"))
    assert set(state.fields.keys()) == {"phi", "T"}
    assert interior_view(state.fields['phi']).shape == (4, 4)


def test_divergence_drops_after_projection():
    state = allocate_state(10, 10)
    # seed random interior velocity
    rng = np.random.default_rng(0)
    interior_view(state.fields['u'])[:] = rng.standard_normal((10, 10)) * 0.01
    interior_view(state.fields['v'])[:] = rng.standard_normal((10, 10)) * 0.01
    tracker = ResidualManager()
    cfg = DummyConfig()
    # Measure pre-step divergence norm
    from pyfoamclone.numerics.fluid_ops import divergence
    pre_div = np.linalg.norm(divergence(state))
    state, residuals, diag = step(cfg, state, tracker, iteration=0)
    post_div = np.linalg.norm(divergence(state))
    assert post_div <= pre_div * 0.95  # expect at least 5% reduction


def test_run_case_manifest(tmp_path):
    cfg_path = tmp_path / 'cfg.json'
    cfg = {
        "schema_version": 1,
        "nx": 6,
        "ny": 6,
        "lx": 1.0,
        "ly": 1.0,
        "Re": 100.0,
        "max_iter": 2,
        "solver": "physical",
        "cfl_target": 0.5,
        "cfl_growth": 1.1,
        "tol": 1e-12
    }
    cfg_path.write_text(json.dumps(cfg))
    state, manifest = run_case(str(cfg_path))
    # Check result has the expected structure
    assert isinstance(state, State), "run_case should return a State object as first item"
    assert isinstance(manifest, dict), "run_case should return a manifest dictionary as second item"
    assert 'config_hash' in manifest, "manifest should have a 'config_hash' key"


def test_cavity_bc_application():
    state = allocate_state(12, 10)
    from pyfoamclone.boundary_conditions.cavity import apply_velocity_bc_cavity
    apply_velocity_bc_cavity(state, lid_velocity=2.5)
    u = interior_view(state.fields['u'])
    v = interior_view(state.fields['v'])
    # Top lid interior excludes corners
    assert np.allclose(u[-1,1:-1], 2.5)
    assert u[-1,0] == 0.0 and u[-1,-1] == 0.0
    assert np.allclose(v[-1,:], 0.0)
    # Bottom and side walls zero
    assert np.allclose(u[0,:], 0.0)
    assert np.allclose(v[0,:], 0.0)
    assert np.allclose(u[:,0], 0.0)
    assert np.allclose(u[:,-1], 0.0)
