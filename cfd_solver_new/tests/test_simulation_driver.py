from pyflow.drivers.simulation_driver import SimulationDriver
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager

class Cfg:
    # minimal config needed by step()
    disable_advection = True
    advection_scheme = 'upwind'
    cfl_target = 0.5
    cfl_growth = 1.05
    Re = 100.0
    lid_velocity = 0.0
    test_mode = False
    lin_tol = 1e-10
    lin_maxiter = 50
    nx = 8
    ny = 8
    lx = nx - 1
    ly = ny - 1


def test_driver_yields_and_advances():
    cfg = Cfg()
    state = allocate_state(cfg.nx, cfg.ny)
    tracker = ResidualManager()
    driver = SimulationDriver(cfg, state, tracker)
    iters = 0
    last_dt = None
    for st, residuals, diag in driver.run(max_steps=5):
        assert st is state  # same state mutated
        assert 'dt' in diag
        if last_dt is not None:
            assert diag['dt'] <= last_dt * 1.5  # basic sanity
        last_dt = diag['dt']
        iters += 1
    assert iters == 5
