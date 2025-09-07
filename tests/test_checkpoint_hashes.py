import json
import numpy as np
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.io.checkpoint import save_checkpoint, load_checkpoint
from pyflow.residuals.manager import ResidualManager
from pyflow.solvers.solver import step
import os

class DummyCfg:
    def __init__(self, nx, ny):
        self.nx = nx; self.ny = ny; self.Re = 100.0
        self.lid_velocity = 1.0
        self.cfl_target = 0.5
        self.cfl_growth = 1.05
        self.advection_scheme = 'upwind'
        self.disable_advection = False
        self.lin_tol = 1e-10
        self.lin_maxiter = 20
        self.diagnostics = False


def test_checkpoint_field_hashes_roundtrip(tmp_path):
    cfg = DummyCfg(12,12)
    st = allocate_state(12,12)
    tracker = ResidualManager()
    for it in range(3):
        st, residuals, diag = step(cfg, st, tracker, it)
    ck = tmp_path / 'ck.npz'
    save_checkpoint(str(ck), st, 3, 0.01, cfg)
    st2, meta = load_checkpoint(str(ck))
    # Recompute hashes from loaded state arrays and compare
    import hashlib
    for name, arr in st2.fields.items():
        h = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
        assert meta['field_hashes'][name] == h


