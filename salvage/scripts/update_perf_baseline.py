import json, os, time
from pathlib import Path
import numpy as np
from framework.state import Mesh, SolverState
from framework.projection_solver import pressure_rhs_unscaled, solve_pressure_poisson_unscaled


def run(method, n=33, repeats=1):
    os.environ['PROJECTION_ENABLE'] = '1'
    os.environ['PROJECTION_LINSOLVER'] = method
    times = []
    its = []
    for _ in range(repeats):
        mesh = Mesh(nx=n, ny=n)
        st = SolverState(mesh=mesh, fields={}, nu=0.01, rho=1.0)
        u = st.require_field('u',(n,n)); v = st.require_field('v',(n,n))
        u[:] = np.random.randn(n,n)
        v[:] = np.random.randn(n,n)
        st.require_field('p',(n,n))
        rhs = pressure_rhs_unscaled(st)
        t0 = time.time()
        it = solve_pressure_poisson_unscaled(rhs, st, tol=1e-5, max_iter=1000)
        times.append(time.time()-t0)
        its.append(it)
    return {'best_time': min(times), 'mean_time': float(np.mean(times)), 'iters': int(np.median(its))}


def main():
    methods = ['jacobi','sor','mg']
    data = {}
    for m in methods:
        try:
            data[m] = run(m)
        except Exception as e:
            data[m] = {'error': str(e)}
    target = Path('perf_baseline.json')
    with target.open('w') as f:
        json.dump(data,f,indent=2)
    print('Wrote', target)

if __name__ == '__main__':
    main()
