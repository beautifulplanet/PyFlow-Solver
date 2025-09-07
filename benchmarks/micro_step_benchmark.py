from __future__ import annotations

"""Micro benchmark: fixed small grid stepping performance.

Usage:
  python -m benchmarks.micro_step_benchmark --steps 50 --nx 32 --ny 32
"""
import argparse, time
from pyflow.config.model import SimulationConfig
from pyflow.core.ghost_fields import allocate_state
from pyflow.residuals.manager import ResidualManager
from pyflow.drivers.simulation_driver import SimulationDriver


def run(steps: int, nx: int, ny: int) -> float:
    cfg = SimulationConfig(nx=nx, ny=ny)
    st = allocate_state(nx, ny)
    drv = SimulationDriver(cfg, st, ResidualManager())
    t0 = time.time()
    gen = drv.run(max_steps=steps)
    for _ in range(steps):
        next(gen)
    return time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    args = ap.parse_args(argv)
    dt = run(args.steps, args.nx, args.ny)
    print(f"BENCH micro_step: steps={args.steps} nx={args.nx} ny={args.ny} elapsed={dt:.3f}s")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())