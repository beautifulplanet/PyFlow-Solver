from pyfoamclone.core.ghost_fields import allocate_state, State
from pyfoamclone.residuals.manager import ResidualManager
from pyfoamclone.solvers.solver import run_loop


class DummyConfig:
    cfl_target = 0.5
    cfl_growth = 1.1
    tol = 1e-12
    dt_prev = 1e-2


def test_run_loop_executes():
    state = allocate_state(4,4)
    assert isinstance(state, State)
    tracker = ResidualManager()
    cfg = DummyConfig()
    state, diag_series = run_loop(cfg, state, 5, tracker)
    assert len(diag_series) >= 1
    assert set(['Ru','Rv','Rp','continuity']).issubset(diag_series[0].keys())