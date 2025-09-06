import math
import numpy as np
from cfd_solver.pyflow.core import load_config, ValidationError
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import diffusion_residual
from cfd_solver.pyflow.core import guard_synthetic, SyntheticUsageError

def test_config_success():
    cfg = load_config({
        'mesh': {'nx': 9, 'ny': 9},
        'physics': {'case': 'diffusion'}
    })
    assert cfg.get('mesh.nx') == 9
    assert cfg.get('mesh.ny') == 9
    assert cfg.get('physics.case') == 'diffusion'


def test_config_missing():
    try:
        load_config({'mesh': {'nx': 9}, 'physics': {}})
    except ValidationError as e:
        assert 'Missing required' in str(e)
    else:
        assert False, 'Expected ValidationError'


def test_state_and_residual():
    m = Mesh(nx=17, ny=17)
    st = SolverState(mesh=m, fields={})
    u = st.require_field('u', (17, 17))
    # Manufactured solution u = sin(pi x) sin(pi y)
    x = np.linspace(0, 1, 17)
    y = np.linspace(0, 1, 17)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
    u[:, :] = u_exact
    f = 2 * math.pi**2 * u_exact  # -Lap(u) = f
    dx = m.dx(); dy = m.dy()
    res = diffusion_residual(u, f, dx, dy)
    assert res['L2'] < 5e-2  # coarse grid tolerance


def test_synthetic_guard_block(monkeypatch):
    monkeypatch.setenv('SYNTHETIC_KILL', '1')
    try:
        guard_synthetic(True, context='unit_test')
    except SyntheticUsageError:
        pass
    else:
        assert False, 'Should have raised SyntheticUsageError'
