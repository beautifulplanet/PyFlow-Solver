import json, os, sys, tempfile, shutil, numpy as np
from pyflow.drivers.simulation_driver import SimulationDriver
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager
from pyflow.io.checkpoint import save_checkpoint, load_checkpoint

class Cfg:
    disable_advection = True
    advection_scheme = 'upwind'
    cfl_target = 0.5
    cfl_growth = 1.05
    Re = 100.0
    lid_velocity = 0.0
    lin_tol = 1e-10
    lin_maxiter = 60
    nx = 12
    ny = 10
    lx = nx - 1
    ly = ny - 1

def run_steps(cfg, steps, state=None, start_it=0):
    tracker = ResidualManager()
    if state is None:
        state = allocate_state(cfg.nx, cfg.ny)
    driver = SimulationDriver(cfg, state, tracker)
    last_diag = None
    for st, res, diag in driver.run(max_steps=steps, start_iteration=start_it):
        last_diag = diag
    return state, last_diag

def test_checkpoint_restart_bitwise_equivalence():
    cfg = Cfg()
    # Full uninterrupted run N+M
    N = 5; M = 4
    full_state, _ = run_steps(cfg, N+M)
    full_u = interior_view(full_state.fields['u']).copy()
    full_v = interior_view(full_state.fields['v']).copy()
    full_p = interior_view(full_state.fields['p']).copy()
    # Run N steps, checkpoint, then restart for M
    part_state, diagN = run_steps(cfg, N)
    with tempfile.TemporaryDirectory() as td:
        ck_path = os.path.join(td, 'ck.npz')
    wall_time = 0.0 if diagN is None else diagN.get('wall_time', 0.0)
    save_checkpoint(ck_path, part_state, N, wall_time, cfg)
        # Load and resume
        loaded, meta = load_checkpoint(ck_path)
        assert meta['iteration'] == N
        resumed_state, _ = run_steps(cfg, M, state=loaded, start_it=N+1)
        ru = interior_view(resumed_state.fields['u'])
        rv = interior_view(resumed_state.fields['v'])
        rp = interior_view(resumed_state.fields['p'])
    # Compare final interior fields (bitwise close)
    assert np.allclose(ru, full_u, rtol=0, atol=1e-12)
    assert np.allclose(rv, full_v, rtol=0, atol=1e-12)
    assert np.allclose(rp, full_p, rtol=0, atol=1e-12)
