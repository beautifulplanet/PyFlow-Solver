import os, sys, subprocess, json
from pathlib import Path
import pytest
from pyflow.config.model import SimulationConfig
from pyflow.core.ghost_fields import allocate_state
from pyflow.io.checkpoint import save_checkpoint
from pyflow.residuals.manager import ResidualManager
from pyflow.drivers.simulation_driver import SimulationDriver


def _run_steps(cfg: SimulationConfig, steps: int = 2):
    state = allocate_state(cfg.nx, cfg.ny)
    drv = SimulationDriver(cfg, state, ResidualManager())
    last = None
    for st, residuals, diag in drv.run(max_steps=steps, start_iteration=0, progress=False, checkpoint_path=None, checkpoint_interval=0):
        last = (st, residuals, diag)
    return last

def test_restart_hash_mismatch_refused(tmp_path):
    # Produce a checkpoint with one config
    cfg_a = SimulationConfig(nx=12, ny=12, Re=150.0)
    st_res = _run_steps(cfg_a, steps=1)
    assert st_res is not None
    state, residuals, diag = st_res
    ck = tmp_path / "ck_a.npz"
    save_checkpoint(str(ck), state, diag['iteration'], diag['wall_time'], cfg_a)
    # Build a different config (hash must differ)
    cfg_b = SimulationConfig(nx=13, ny=12, Re=150.0)
    assert cfg_a.config_hash != cfg_b.config_hash
    # Invoke CLI via module to attempt restart; expect refusal (exit code 3)
    src_path = Path(__file__).resolve().parents[1] / 'src'
    cmd = [sys.executable, '-m', 'pyflow.cli', 'run', '--nx', str(cfg_b.nx), '--ny', str(cfg_b.ny), '--re', str(cfg_b.Re), '--restart', str(ck), '--steps', '2']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(src_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert 'Refusing restart' in (proc.stdout + proc.stderr)

def test_restart_hash_override_allows(tmp_path):
    cfg_a = SimulationConfig(nx=12, ny=12, Re=150.0)
    st_res = _run_steps(cfg_a, steps=1)
    assert st_res is not None
    state, residuals, diag = st_res
    ck = tmp_path / "ck_a.npz"
    save_checkpoint(str(ck), state, diag['iteration'], diag['wall_time'], cfg_a)
    cfg_b = SimulationConfig(nx=13, ny=12, Re=150.0)
    assert cfg_a.config_hash != cfg_b.config_hash
    src_path = Path(__file__).resolve().parents[1] / 'src'
    cmd = [sys.executable, '-m', 'pyflow.cli', 'run', '--nx', str(cfg_b.nx), '--ny', str(cfg_b.ny), '--re', str(cfg_b.Re), '--restart', str(ck), '--allow-hash-mismatch', '--steps', '2']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(src_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Should run normally (exit 0) or early exit 0 if simulation completes
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_cli_validate_outputs_hash():
    src_path = Path(__file__).resolve().parents[1] / 'src'
    cmd = [sys.executable, '-m', 'pyflow.cli', 'validate', '--nx', '16', '--ny', '12']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(src_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert 'config_hash' in data
    assert data['nx'] == 16 and data['ny'] == 12