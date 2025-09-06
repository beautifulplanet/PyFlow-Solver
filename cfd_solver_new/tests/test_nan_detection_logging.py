import os, json, tempfile, numpy as np
from types import SimpleNamespace
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager
from pyflow.drivers.simulation_driver import SimulationDriver

class Cfg(SimpleNamespace):
    pass

def test_nan_detection_and_logging():
    cfg = Cfg(disable_advection=True, advection_scheme='upwind', cfl_target=0.5, cfl_growth=1.05,
              Re=100.0, lid_velocity=0.0, lin_tol=1e-10, lin_maxiter=30, nx=10, ny=10, lx=9, ly=9,
              diagnostics=False)
    state = allocate_state(cfg.nx, cfg.ny)
    # Inject NaN into interior u field to trigger detection
    ui = interior_view(state.fields['u'])
    ui[2,2] = np.nan
    with tempfile.TemporaryDirectory() as td:
        log_path = os.path.join(td, 'run.log')
        setattr(cfg, 'log_path', log_path)
        tracker = ResidualManager()
        driver = SimulationDriver(cfg, state, tracker)
        # Run one step; detection happens in step()
        for _st, _res, diag in driver.run(max_steps=1):
            assert diag['nan_detected'] is True
        # Log file should contain an error entry
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        assert any(json.loads(l).get('type')=='error' for l in lines)