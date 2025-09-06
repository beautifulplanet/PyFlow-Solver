"""Manufactured solution convergence test for 2D Poisson equation.

Solves: - (u_xx + u_yy) = f on (0,1)x(0,1) with Dirichlet boundaries
Exact solution chosen: u(x,y) = sin(pi x) * sin(pi y)
Then f(x,y) = 2 * pi^2 * sin(pi x) * sin(pi y)

Discretization: 5-point second-order finite difference on uniform grid.
Solver: Simple Jacobi iterative scheme with damping (omega) until residual L_inf threshold.
Outputs: Printed convergence table + JSON file (poisson_ms_results.json) with errors vs grid size.

Rationale:
Validates presence of a genuinely solved PDE (not synthetic interpolation) and demonstrates
second-order spatial convergence (error ~ O(h^2)). Provides an anchor test for future core solver.

Quick integration guidance:
- Treat this as an initial physics correctness gate; integrate into CI nightly run.
- Extend later with: variable coefficients, Neumann sides, different manufactured solutions.
"""
from __future__ import annotations
import json
import os
import math
import time
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from framework.logging import init_run

@dataclass
class PoissonResult:
    nx: int
    ny: int
    iterations: int
    runtime_s: float
    linf_residual: float
    l2_error: float
    linf_error: float
    h: float  # representative spacing (assuming hx ~ hy)


def exact_u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x) * np.sin(math.pi * y)


def rhs_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y)


def solve_poisson_ms(nx: int, ny: int, tol: float = 1e-8, max_iter: int = 50_000, omega: float = 1.0, progress: bool = False, progress_every: int = 500) -> PoissonResult:
    assert nx >= 5 and ny >= 5, "Need at least 5 points each direction for sensible stencil"
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (hx2 + hy2)

    # 2D grid
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_exact = exact_u(X, Y)
    f = rhs_f(X, Y)

    # Initialize solution array including boundaries
    u = np.zeros_like(u_exact)

    # Apply Dirichlet BC from exact solution at boundary
    u[0, :] = u_exact[0, :]
    u[-1, :] = u_exact[-1, :]
    u[:, 0] = u_exact[:, 0]
    u[:, -1] = u_exact[:, -1]

    residual_linf = float('inf')
    it = 0
    start = time.time()

    # Jacobi buffers (avoid in-place Gauss-Seidel to keep method simple & vectorizable)
    u_new = u.copy()
    # NOTE: Still Jacobi (not Gauss-Seidel / multigrid). Adequate for manufactured convergence gate.
    # Future: replace with better smoother (SOR / multigrid V-cycle) to cut iterations dramatically.
    residual_prev = None
    while residual_linf > tol and it < max_iter:
        # Interior update (vectorized)
        # u_new[i,j] = ((u[i+1,j] + u[i-1,j]) * hy2 + (u[i,j+1] + u[i,j-1]) * hx2 - f[i,j] * hx2 * hy2) / denom
        u_new[1:-1, 1:-1] = (
            (u[2:, 1:-1] + u[:-2, 1:-1]) * hy2 +
            (u[1:-1, 2:] + u[1:-1, :-2]) * hx2 +
            f[1:-1, 1:-1] * hx2 * hy2
        ) / denom

        # Damped update
        u_new[1:-1, 1:-1] = omega * u_new[1:-1, 1:-1] + (1 - omega) * u[1:-1, 1:-1]

        # Residual (discrete Laplacian + f) on interior
        lap_u = (
            (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]) / hx2 +
            (u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2]) / hy2
        )
        res = -lap_u - f[1:-1, 1:-1]
        residual_linf = float(np.max(np.abs(res)))
        if residual_prev is not None and residual_prev / max(residual_linf, 1e-300) < 1.0001:
            break
        residual_prev = residual_linf
        if progress and (it % progress_every == 0 or residual_linf <= tol):
            print(f"  iter={it:6d} resid={residual_linf:.3e}")

        u, u_new = u_new, u  # swap references
        it += 1

    runtime = time.time() - start

    # Error metrics (include all interior points including boundaries for global norm)
    diff = u - u_exact
    l2_error = float(np.sqrt(np.mean(diff**2)))
    linf_error = float(np.max(np.abs(diff)))

    return PoissonResult(
        nx=nx,
        ny=ny,
        iterations=it,
        runtime_s=runtime,
        linf_residual=residual_linf,
        l2_error=l2_error,
        linf_error=linf_error,
        h=max(hx, hy),
    )


def run_suite(grids: List[int] | None = None, tol: float = 1e-8, omega: float = 1.0, progress: bool = False, run_logger=None) -> Dict[str, List[Dict[str, float]]]:
    if grids is None:
        grids = [17, 33, 65]  # trimmed default for fast CI sanity (can extend locally to 129, 257, ...)
    print(f"Running Poisson manufactured suite: grids={grids} tol={tol} omega={omega}")
    results: List[PoissonResult] = []
    for n in grids:
        r = solve_poisson_ms(n, n, tol=tol, omega=omega, progress=progress)
        results.append(r)
        print(f"N={n:3d} iter={r.iterations:5d} h={r.h:.5f} L2={r.l2_error:.3e} Linf={r.linf_error:.3e} Rres={r.linf_residual:.3e} t={r.runtime_s:.2f}s")
        if run_logger:
            run_logger.log_event("INFO", "grid_complete", {"N": n, "iterations": r.iterations, "l2_error": r.l2_error, "linf_error": r.linf_error, "residual": r.linf_residual})

    # Estimate observed order p using successive pairs (p ~ log(e_i/e_{i+1}) / log(h_i/h_{i+1}))
    orders = []
    for i in range(len(results) - 1):
        e1, e2 = results[i].l2_error, results[i+1].l2_error
        h1, h2 = results[i].h, results[i+1].h
        p = math.log(e1 / e2) / math.log(h1 / h2)
        orders.append(p)

    if orders:
        print("Observed L2 order (pairwise):", ", ".join(f"{o:.2f}" for o in orders))

    data = [r.__dict__ for r in results]
    out = {"results": data, "pairwise_L2_order": orders}
    with open("poisson_ms_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if orders:
        print(f"Final Summary: min_order={min(orders):.2f} max_order={max(orders):.2f} finest_residual={results[-1].linf_residual:.3e}")
    if run_logger:
        run_logger.log_event("INFO", "suite_complete", {"min_order": min(orders) if orders else None, "max_order": max(orders) if orders else None})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Manufactured Poisson convergence test")
    ap.add_argument("--grids", type=int, nargs="*", help="Grid sizes (single integer implies square grid). Default: 17 33 65")
    ap.add_argument("--tol", type=float, default=1e-8, help="Residual L_inf tolerance (default 1e-8)")
    ap.add_argument("--omega", type=float, default=1.0, help="Jacobi damping factor (<=1). Future SOR not yet implemented.")
    ap.add_argument("--progress", action="store_true", help="Print periodic residual progress")
    ap.add_argument("--ci-gate", action="store_true", help="Fail (exit 1) if any pairwise order < threshold or final residual > tol")
    ap.add_argument("--order-threshold", type=float, default=1.8, help="Minimum acceptable pairwise L2 order (default 1.8)")
    ap.add_argument("--metrics-out", type=str, default="poisson_perf_metrics.json", help="Performance metrics JSON output path")
    args = ap.parse_args()
    grids = args.grids if args.grids else None
    run_logger = init_run(project_name=os.environ.get('PROJECT_NAME','cfdmini'), log_dir='logs')
    out = run_suite(grids=grids, tol=args.tol, omega=args.omega, progress=args.progress, run_logger=run_logger)
    # Write performance metrics
    perf = [
        {
            "nx": r["nx"],
            "ny": r["ny"],
            "iterations": r["iterations"],
            "runtime_s": r["runtime_s"],
            "linf_residual": r["linf_residual"],
            "l2_error": r["l2_error"],
            "linf_error": r["linf_error"],
            "h": r["h"],
        } for r in out["results"]
    ]
    try:
        # Soft regression note: iterations should generally increase with grid size for plain Jacobi
        iters = [p["iterations"] for p in perf]
        monotonic_non_decreasing = all(iters[i] <= iters[i+1] for i in range(len(iters)-1))
        meta = {"monotonic_iterations": monotonic_non_decreasing}
        with open(args.metrics_out, "w", encoding="utf-8") as mf:
            json.dump({"grids": perf, "meta": meta}, mf, indent=2)
    except Exception as e:
        print(f"WARNING: could not write metrics file: {e}")

    if args.ci_gate:
        import sys
        orders = out["pairwise_L2_order"]
        if not orders:
            print("CI GATE FAIL: insufficient grids for order computation")
            sys.exit(1)
        failing = [i for i,o in enumerate(orders) if o < args.order_threshold]
        if failing:
            print(f"CI GATE FAIL: pair indices {failing} below threshold {args.order_threshold} (orders={orders})")
            sys.exit(1)
        # Check residual from finest grid result
        finest = out["results"][-1]
        if finest["linf_residual"] > args.tol:
            print(f"CI GATE FAIL: residual {finest['linf_residual']:.3e} > tol {args.tol:.3e}")
            sys.exit(1)
        print(f"CI GATE PASS (min_order={min(orders):.2f} all_pairs>={args.order_threshold} finest_residual={finest['linf_residual']:.3e})")
    run_logger.log_event("INFO", "ci_gate", {"status": "pass" if (not args.ci_gate or (orders and min(orders) >= args.order_threshold)) else "fail"})
    run_logger.close()
