"""Projection performance harness.

Measures wall time and iterations for Jacobi vs SOR on selected grids and
writes JSON baseline (or compares to existing baseline).

Usage (record baseline):
  PROJECTION_ENABLE=1 python scripts/perf_projection.py --baseline out/perf_projection_baseline.json

Usage (compare):
  PROJECTION_ENABLE=1 python scripts/perf_projection.py --compare out/perf_projection_baseline.json
"""
from __future__ import annotations
import argparse, json, time, os, math
from pathlib import Path
import numpy as np
from framework.state import Mesh, SolverState
from framework.projection_solver import projection_step


def make_state(n: int) -> SolverState:
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01, rho=1.0)
    u = st.require_field('u', (n,n))
    v = st.require_field('v', (n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u[:,:] = np.sin(math.pi*X)
    v[:,:] = np.cos(math.pi*Y)
    st.require_field('p', (n,n))
    return st


def run_once(n: int, method: str, dt: float = 2e-4):
    os.environ['PROJECTION_LINSOLVER'] = method
    st = make_state(n)
    t0 = time.perf_counter()
    stats = projection_step(st, dt=dt, use_advection=False, adaptive_dt=False)
    t1 = time.perf_counter()
    return {
        'grid': n,
        'method': method,
        'time_s': t1 - t0,
        'iters': stats.iters,
        'div_before': stats.notes['div_linf_before'],
        'div_after': stats.notes['div_linf']
    }


def measure(grids, repeats):
    rows = []
    for n in grids:
        for method, desc in METHODS:
            best = None
            for _ in range(repeats):
                r = run_once(n, method)
                if best is None or r['time_s'] < best['time_s']:
                    best = r
            rows.append(best)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', type=int, nargs='+', default=[33,65])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--baseline', type=str, help='Write baseline JSON path')
    ap.add_argument('--compare', type=str, help='Compare against existing baseline')
    ap.add_argument('--tolerance', type=float, default=0.15, help='Allowed relative slowdown vs baseline')
    args = ap.parse_args()

    if not (args.baseline or args.compare):
        ap.error('Specify --baseline or --compare')

    os.environ['PROJECTION_ENABLE'] = '1'
    rows = measure(args.grids, args.repeats)
    if args.baseline:
        Path(args.baseline).parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline,'w') as f:
            json.dump({'results': rows, 'schema':'projection_perf_v1'}, f, indent=2)
        print(f'Baseline written: {args.baseline}')
        return
    # Compare path
    with open(args.compare) as f:
        base = json.load(f)
    base_map = {(r['grid'], r['method']): r for r in base['results']}
    failures = []
    for r in rows:
        key = (r['grid'], r['method'])
        if key not in base_map:
            print(f'[WARN] Missing baseline entry for {key}')
            continue
        b = base_map[key]
        rel = (r['time_s'] - b['time_s']) / b['time_s']
        print(f"{key} time {r['time_s']:.4f}s (base {b['time_s']:.4f}s) rel={rel:.1%} iters {r['iters']} (base {b['iters']})")
        if rel > args.tolerance:
            failures.append((key, rel))
    if failures:
        raise SystemExit(f'Performance regressions: {failures}')
    print('Performance within tolerance.')

METHODS = [
    ('jacobi', 'baseline Jacobi'),
    ('sor', 'successive over-relaxation'),
    ('mg', 'multigrid V-cycle'),
]

if __name__ == '__main__':
    main()
