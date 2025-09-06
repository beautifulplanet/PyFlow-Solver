from __future__ import annotations
"""Project PyFlow-Measure: Benchmark Harness.

Purpose:
    Provide a reproducible measurement tool for baseline performance
    (before any optimization work) across multiple grid sizes for the
    lid-driven cavity configuration.

Metrics Captured per Grid Size:
    - total_time_s: wall clock from first step start to last step completion
    - avg_time_per_step_s: total_time_s / steps
    - avg_cg_iterations: mean pressure CG iterations (Rp_iterations) per step

Usage (module):
    python -m pyflow.benchmark.harness --grids 32,64,128,256 --steps 100 --json-out bench.json

Design Notes:
    * Keeps configuration minimal (advection disabled to isolate projection costs initially).
    * Suppresses verbose prints to avoid skewing timing.
    * Uses existing SimulationDriver for consistency.
    * Does NOT attempt micro-optimizations; purely observational.
"""
import argparse, json, time
from types import SimpleNamespace
from typing import List, Dict, Any

from ..drivers.simulation_driver import SimulationDriver
from ..core.ghost_fields import allocate_state
from ..residuals.manager import ResidualManager

def make_config(nx: int, ny: int, *, Re: float = 100.0, lid_velocity: float = 0.0, disable_advection: bool = True, enable_jacobi_pc: bool = True):
    # Minimal config object with attributes consumed by step() / pressure solver
    cfg = SimpleNamespace(
        nx=nx, ny=ny,
        Re=Re,
        lid_velocity=lid_velocity,
        cfl_target=0.5,
        cfl_growth=1.05,
        advection_scheme='upwind',
        disable_advection=disable_advection,
        lin_tol=1e-10,
        lin_maxiter=400,
        diagnostics=False,
        enable_jacobi_pc=enable_jacobi_pc,
        force_quiet=True,
        lx=nx - 1,
        ly=ny - 1,
    )
    return cfg

def run_case(nx: int, ny: int, steps: int) -> Dict[str, Any]:
    cfg = make_config(nx, ny)
    state = allocate_state(nx, ny)
    tracker = ResidualManager()
    driver = SimulationDriver(cfg, state, tracker)
    t0 = time.time()
    total_cg_iter = 0
    step_count = 0
    for _s, _res, diag in driver.run(max_steps=steps):
        total_cg_iter += int(diag.get('Rp_iterations', 0))
        step_count += 1
    total_time = time.time() - t0
    avg_time = total_time / max(step_count, 1)
    avg_cg = total_cg_iter / max(step_count, 1)
    return {
        'nx': nx,
        'ny': ny,
        'steps': step_count,
        'total_time_s': total_time,
        'avg_time_per_step_s': avg_time,
        'avg_cg_iterations': avg_cg,
    }

def benchmark(grids: List[int], steps: int) -> Dict[str, Any]:
    runs = [run_case(n, n, steps) for n in grids]
    return {
        'benchmark': 'lid_driven_cavity',
        'steps_requested': steps,
        'runs': runs,
    }

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='PyFlow Benchmark Harness')
    ap.add_argument('--grids', type=str, default='32,64,128,256', help='Comma-separated grid sizes (nx=ny)')
    ap.add_argument('--steps', type=int, default=100, help='Number of steps per grid')
    ap.add_argument('--json-out', type=str, default=None, help='Path to write JSON results')
    args = ap.parse_args(argv)
    grids = [int(x) for x in args.grids.split(',') if x.strip()]
    result = benchmark(grids, args.steps)
    payload = json.dumps(result, indent=2)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            f.write(payload + '\n')
    print(payload)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())

__all__ = ["benchmark", "run_case", "make_config"]