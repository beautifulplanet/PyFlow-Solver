"""Playground: Manufactured Solution Order Verification (v0.1)

Purpose: Empirically estimate spatial order of accuracy for prototype gradient and Laplacian operators on a uniform 2D grid using a smooth manufactured solution.

No external solver dependencies; numpy only.
"""
from __future__ import annotations
import argparse, json, math, os, statistics
from dataclasses import dataclass
import numpy as np

SCHEMA_VERSION = 1

@dataclass
class Result:
    n: int
    h: float
    grad_l2_error: float
    lap_l2_error: float


def manufactured_u(x, y):
    return np.sin(math.pi * x) * np.sin(math.pi * y)

def exact_grad(x, y):
    du_dx = math.pi * np.cos(math.pi * x) * np.sin(math.pi * y)
    du_dy = math.pi * np.sin(math.pi * x) * np.cos(math.pi * y)
    return du_dx, du_dy

def exact_lap(x, y):
    # Laplacian of sin(pi x) sin(pi y) = -2 pi^2 sin(pi x) sin(pi y)
    return -2 * (math.pi ** 2) * manufactured_u(x, y)

def compute_gradient(u, h):
    # Central differences interior, 2nd order; first-order one-sided at boundary
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dudx[:,1:-1] = (u[:,2:] - u[:,0:-2]) / (2*h)
    dudy[1:-1,:] = (u[2:,:] - u[0:-2,:]) / (2*h)
    # boundaries
    dudx[:,0] = (u[:,1] - u[:,0]) / h
    dudx[:,-1] = (u[:,-1] - u[:,-2]) / h
    dudy[0,:] = (u[1,:] - u[0,:]) / h
    dudy[-1,:] = (u[-1,:] - u[-2,:]) / h
    return dudx, dudy

def compute_laplacian(u, h):
    lap = np.zeros_like(u)
    lap[1:-1,1:-1] = (u[1:-1,2:] + u[1:-1,0:-2] + u[2:,1:-1] + u[0:-2,1:-1] - 4*u[1:-1,1:-1]) / (h*h)
    # naive second-order ignoring boundary special treatment
    return lap

def l2_error(field, exact):
    return math.sqrt(np.mean((field - exact)**2))

def run(grids):
    results=[]
    for n in grids:
        h = 1.0/(n-1)
        xs = np.linspace(0,1,n)
        ys = np.linspace(0,1,n)
        X,Y = np.meshgrid(xs, ys)
        U = manufactured_u(X,Y)
        dudx_exact = np.zeros_like(U); dudy_exact = np.zeros_like(U)
        lap_exact = np.zeros_like(U)
        for i in range(n):
            for j in range(n):
                gx, gy = exact_grad(X[i,j], Y[i,j])
                dudx_exact[i,j] = gx
                dudy_exact[i,j] = gy
                lap_exact[i,j] = exact_lap(X[i,j], Y[i,j])
        dudx_num, dudy_num = compute_gradient(U, h)
        lap_num = compute_laplacian(U, h)
        grad_err = l2_error(dudx_num, dudx_exact) + l2_error(dudy_num, dudy_exact)
        lap_err = l2_error(lap_num, lap_exact)
        results.append(Result(n=n, h=h, grad_l2_error=grad_err, lap_l2_error=lap_err))
    return results

def estimate_order(results, attr):
    # log-log least squares
    xs = [math.log(r.h) for r in results]
    ys = [math.log(getattr(r, attr)) for r in results]
    xbar = statistics.mean(xs); ybar = statistics.mean(ys)
    num = sum((x - xbar)*(y - ybar) for x,y in zip(xs,ys))
    den = sum((x - xbar)**2 for x in xs)
    slope = num/den if den else float('nan')
    return slope  # slope ~ order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grids', type=int, nargs='+', default=[17,33,65,129])
    ap.add_argument('--output-dir', default='playground/artifacts/manufactured')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results = run(args.grids)
    grad_order = estimate_order(results, 'grad_l2_error')
    lap_order = estimate_order(results, 'lap_l2_error')
    summary = {
        'schema_version': SCHEMA_VERSION,
        'grids': args.grids,
        'grad_estimated_order': -grad_order,  # negative slope
        'lap_estimated_order': -lap_order,
        'points': [r.__dict__ for r in results]
    }
    with open(os.path.join(args.output_dir, 'manufactured_results.json'),'w') as f:
        json.dump(summary, f, indent=2)
    for r in results:
        print(json.dumps({'n': r.n, 'h': r.h, 'grad_err': r.grad_l2_error, 'lap_err': r.lap_l2_error}))

if __name__ == '__main__':
    main()
