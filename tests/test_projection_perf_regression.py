import json, os, math, time
import numpy as np
import pytest
from pathlib import Path
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step

BASELINE_PATH = Path('perf_baseline.json')


def make_state(n):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01, rho=1.0)
    u = st.require_field('u',(n,n)); v = st.require_field('v',(n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u[:,:] = np.sin(math.pi*X); v[:,:] = np.cos(math.pi*Y)
    st.require_field('p',(n,n))
    return st

@pytest.mark.slow
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason='Baseline JSON absent; generate with scripts/perf_projection.py --baseline')
def test_projection_perf_regression():
    os.environ['PROJECTION_ENABLE'] = '1'
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    # Support both flat dict (old) and list-of-results (future)
    if 'results' in baseline:
        # New format: list of dicts with grid/method/time/iters
        base_map = {(r['grid'], r['method']): r for r in baseline['results']}
        tolerance = 0.20  # 20% slowdown allowed in CI noise
        for (grid, method), base in base_map.items():
            os.environ['PROJECTION_LINSOLVER'] = method
            st = make_state(grid)
            t0 = time.perf_counter()
            stats = projection_step(st, dt=2e-4, use_advection=False, adaptive_dt=False)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            rel = (elapsed - base['time_s']) / base['time_s']
            assert rel <= tolerance, f"Perf regression grid={grid} method={method} elapsed={elapsed:.4f}s base={base['time_s']:.4f}s rel={rel:.1%}"
            assert stats.iters <= base['iters'] * 1.2 + 10, f"Iteration regression grid={grid} method={method} iters={stats.iters} base={base['iters']}"
    else:
        # Flat dict: method -> {best_time, mean_time, iters}
        for method, entry in baseline.items():
            os.environ['PROJECTION_LINSOLVER'] = method
            st = make_state(33)  # grid size matches baseline script
            t0 = time.perf_counter()
            stats = projection_step(st, dt=2e-4, use_advection=False, adaptive_dt=False)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            rel = (elapsed - entry['best_time']) / entry['best_time']
            assert rel <= 0.20, f"Perf regression method={method} elapsed={elapsed:.4f}s base={entry['best_time']:.4f}s rel={rel:.1%}"
            assert stats.iters <= entry['iters'] * 1.2 + 10, f"Iteration regression method={method} iters={stats.iters} base={entry['iters']}"
