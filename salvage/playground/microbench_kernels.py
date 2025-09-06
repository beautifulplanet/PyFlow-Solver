"""Playground: Microkernel Benchmarks (v0.1)
Benchmarks core numerical primitives (gradient, divergence, laplacian) using vectorized numpy operations.
"""
from __future__ import annotations
import argparse, json, os, time, math
import numpy as np

SCHEMA_VERSION=1

def bench_gradient(nx, ny, reps):
    u=np.random.rand(ny,nx)
    h=1.0/(nx-1)
    start=time.perf_counter()
    for _ in range(reps):
        dudx=(u[:,2:]-u[:,:-2])/(2*h)
        dudy=(u[2:,:]-u[:-2,:])/(2*h)
    end=time.perf_counter()
    cells=nx*ny
    return {'kernel':'gradient','nx':nx,'ny':ny,'reps':reps,'time_s':end-start,'time_per_cell_us':1e6*(end-start)/(cells*reps)}

def bench_laplacian(nx, ny, reps):
    u=np.random.rand(ny,nx)
    h2=(1.0/(nx-1))**2
    start=time.perf_counter()
    for _ in range(reps):
        lap = (u[:,2:]+u[:,:-2]+u[2:,:]+u[:-2,:]-4*u[:,1:-1])/h2
    end=time.perf_counter()
    cells=nx*ny
    return {'kernel':'laplacian','nx':nx,'ny':ny,'reps':reps,'time_s':end-start,'time_per_cell_us':1e6*(end-start)/(cells*reps)}

def bench_divergence(nx, ny, reps):
    u=np.random.rand(ny,nx); v=np.random.rand(ny,nx)
    h=1.0/(nx-1)
    start=time.perf_counter()
    for _ in range(reps):
        div = (u[:,2:]-u[:,:-2])/(2*h) + (v[2:,:]-v[:-2,:])/(2*h)
    end=time.perf_counter()
    cells=nx*ny
    return {'kernel':'divergence','nx':nx,'ny':ny,'reps':reps,'time_s':end-start,'time_per_cell_us':1e6*(end-start)/(cells*reps)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[64,128,256,512])
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--output-dir', default='playground/artifacts/microbench')
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results=[]
    for n in args.sizes:
        for fn in (bench_gradient, bench_laplacian, bench_divergence):
            res=fn(n,n,args.reps)
            results.append(res)
            print(json.dumps(res))
    summary={'schema_version':SCHEMA_VERSION,'results':results}
    with open(os.path.join(args.output_dir,'microbench_results.json'),'w') as f:
        json.dump(summary,f,indent=2)

if __name__=='__main__':
    main()
