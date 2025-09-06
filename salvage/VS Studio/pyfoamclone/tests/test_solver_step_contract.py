import numpy as np
from pyfoamclone.core.ghost_fields import allocate_state, State
from pyfoamclone.residuals.manager import ResidualManager
from pyfoamclone.solvers.solver import step


class DummyConfig:
    cfl_target = 0.5
    cfl_growth = 1.1
    dt_prev = 1e-2


def test_step_contract_and_residual_keys():
    state = allocate_state(6, 5)
    assert isinstance(state, State)
    tracker = ResidualManager()
    cfg = DummyConfig()
    state, residuals, diagnostics = step(cfg, state, tracker, iteration=0)
    assert set(residuals.keys()) == {'Ru','Rv','Rp','continuity'}
    assert 'iteration' in diagnostics and 'dt' in diagnostics and 'CFL' in diagnostics
    # residuals should be floats
    assert all(isinstance(v, float) for v in residuals.values())


def test_step_reports_pressure_diagnostics():
    state = allocate_state(6, 5)
    tracker = ResidualManager()
    cfg = DummyConfig()
    state, residuals, diagnostics = step(cfg, state, tracker, iteration=0)
    assert 'Rp_iterations' in diagnostics
    assert 'Rp_residual' in diagnostics
    assert isinstance(diagnostics['Rp_iterations'], (int, float))
    assert isinstance(diagnostics['Rp_residual'], float)
    assert diagnostics['Rp_iterations'] > 0 or diagnostics['Rp_residual'] >= 0.0
