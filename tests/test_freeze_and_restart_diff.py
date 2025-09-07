from __future__ import annotations

import json, tempfile, os
import numpy as np
import pytest

from pyflow.config.model import SimulationConfig, freeze_config, config_hash
from pyflow.core.ghost_fields import allocate_state
from pyflow.residuals.manager import ResidualManager
from pyflow.drivers.simulation_driver import SimulationDriver
from pyflow.io.checkpoint import save_checkpoint, load_checkpoint


def _one_step(cfg):
    st = allocate_state(cfg.nx, cfg.ny)
    drv = SimulationDriver(cfg, st, ResidualManager())
    gen = drv.run(max_steps=1)
    next(gen)
    return st


def test_freeze_blocks_semantic_mutation():
    cfg = SimulationConfig(nx=8, ny=8)
    freeze_config(cfg)
    with pytest.raises(AttributeError):
        cfg.nx = 10  # type: ignore
    # runtime field allowed
    cfg.force_quiet = True  # type: ignore


def test_restart_diff_core_config_snapshot(capsys):
    cfg = SimulationConfig(nx=8, ny=8)
    st = _one_step(cfg)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ck.npz')
        save_checkpoint(path, st, 0, 0.0, cfg)
        # mutate semantic param by creating new cfg
        cfg2 = SimulationConfig(nx=10, ny=8)
        # Simulate CLI enforcement snippet
        loaded_state, meta = load_checkpoint(path)
        assert meta['config_hash'] != config_hash(cfg2)
        # Show diff printing (simulate)
        # Not asserting output content fully; just ensure core_config present
        assert 'core_config' in meta