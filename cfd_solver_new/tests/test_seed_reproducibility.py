import numpy as np
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager
from pyflow.solvers.solver import step

class DummyCfg:
    def __init__(self, nx, ny, seed):
        self.nx = nx; self.ny = ny; self.Re = 100.0
        self.lid_velocity = 1.0
        self.cfl_target = 0.5
        self.cfl_growth = 1.05
        self.advection_scheme = 'upwind'
        self.disable_advection = False
        self.lin_tol = 1e-10
        self.lin_maxiter = 50
        self.seed = seed
        self.diagnostics = False


def run_once(seed):
    np.random.seed(seed)
    cfg = DummyCfg(16,16,seed)
    st = allocate_state(16,16)
    tracker = ResidualManager()
    for it in range(5):
        st, residuals, diag = step(cfg, st, tracker, it)
    return interior_view(st.fields['u']).copy(), interior_view(st.fields['v']).copy()


def test_seed_repro_produces_identical_fields():
    u1,v1 = run_once(42)
    u2,v2 = run_once(42)
    assert np.allclose(u1,u2)
    assert np.allclose(v1,v2)

