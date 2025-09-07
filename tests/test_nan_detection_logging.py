import io, json, numpy as np
from types import SimpleNamespace
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager
from pyflow.drivers.simulation_driver import SimulationDriver

class Cfg(SimpleNamespace):
    pass

def test_nan_detection_in_memory_logging():
    buf = io.StringIO()
    cfg = Cfg(disable_advection=True, advection_scheme='upwind', cfl_target=0.5, cfl_growth=1.05,
              Re=100.0, lid_velocity=0.0, lin_tol=1e-10, lin_maxiter=30, nx=10, ny=10, lx=9, ly=9,
              diagnostics=False, log_stream=buf)
    state = allocate_state(cfg.nx, cfg.ny)
    interior_view(state.fields['u'])[2,2] = np.nan
    tracker = ResidualManager()
    driver = SimulationDriver(cfg, state, tracker)
    for _st, _res, diag in driver.run(max_steps=1):
        assert diag.get('nan_detected') is True
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    assert any('"reason":"nan_detected"' in l for l in lines), 'Expected nan_detected error log'
