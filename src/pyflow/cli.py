from __future__ import annotations
"""PyFlow Phase 2 CLI – control layer over SimulationDriver.

Adds:
    * Core parameter control (grid size, Reynolds number, lid velocity, steps).
    * Verbosity toggle (diagnostics) to suppress solver internal prints.
    * JSON streaming (--json-stream) producing one structured line per step for
        downstream dashboard ingestion.

Streaming JSON line schema (stable draft v1):
    {
        "type": "step",
        "iteration": int,
        "dt": float,
        "CFL": float,
        "continuity": float,
        "wall_time": float,
        "residuals": {Ru,Rv,Rp,continuity},
        "diagnostics": {... solver extras ...}
    }
"""
import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

from .drivers.simulation_driver import SimulationDriver
from .core.ghost_fields import allocate_state
from .residuals.manager import ResidualManager
from .config.validation import validate_config
from .config.model import config_hash, EXCLUDED_RUNTIME_FIELDS, freeze_config, config_core_dict

@dataclass
class Config:
    nx: int
    ny: int
    Re: float = 100.0
    lid_velocity: float = 1.0
    cfl_target: float = 0.5
    cfl_growth: float = 1.05
    advection_scheme: str = "quick"
    disable_advection: bool = False
    lin_tol: float = 1e-10
    lin_maxiter: int = 300
    diagnostics: bool = False  # suppress verbose step() prints unless requested
    lx: Optional[float] = None
    ly: Optional[float] = None

    def __post_init__(self):
        if self.lx is None:
            self.lx = float(self.nx - 1)
        if self.ly is None:
            self.ly = float(self.ny - 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a PyFlow simulation")
    p.add_argument("--nx", type=int, default=64, help="Interior grid cells in x (default 64)")
    p.add_argument("--ny", type=int, default=64, help="Interior grid cells in y (default 64)")
    p.add_argument("--re", type=float, default=100.0, help="Reynolds number")
    p.add_argument("--lid-velocity", type=float, default=1.0, help="Lid (top) velocity")
    p.add_argument("--steps", "--max-steps", type=int, default=100, help="Maximum solver steps")
    p.add_argument("--scheme", choices=["quick", "upwind"], default="quick", help="Advection scheme")
    p.add_argument("--disable-advection", action="store_true", help="Disable advection predictor stage")
    p.add_argument("--cfl", type=float, default=0.5, help="Target CFL")
    p.add_argument("--cfl-growth", type=float, default=1.05, help="CFL growth limiter factor")
    p.add_argument("--lin-tol", type=float, default=1e-10, help="Linear solver tolerance")
    p.add_argument("--lin-maxiter", type=int, default=300, help="Linear solver max iterations")
    p.add_argument("--json-stream", action="store_true", help="Emit per-step JSON lines (dashboard feed)")
    p.add_argument("--continuity-threshold", type=float, default=None, help="Stop early if continuity residual below this value")
    p.add_argument("--progress", action="store_true", help="Print per-step compact progress line (ignored with --json-stream)")
    p.add_argument("--diagnostics", action="store_true", help="Enable verbose internal solver diagnostics printouts")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to periodic checkpoint (.npz)")
    p.add_argument("--checkpoint-interval", type=int, default=0, help="Write checkpoint every N iterations (0=off)")
    p.add_argument("--no-preconditioner", action="store_true", help="Disable automatic Jacobi preconditioner")
    p.add_argument("--log-jsonl", type=str, default=None, help="Path to structured JSONL log file")
    p.add_argument("--restart", type=str, default=None, help="Checkpoint file to restart from")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--assert-invariants", action="store_true", help="Enable runtime solver invariant assertions")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Config(
        nx=args.nx,
        ny=args.ny,
        Re=args.re,
        lid_velocity=args.lid_velocity,
        cfl_target=args.cfl,
        cfl_growth=args.cfl_growth,
        advection_scheme=args.scheme,
        disable_advection=args.disable_advection,
        lin_tol=args.lin_tol,
        lin_maxiter=args.lin_maxiter,
        diagnostics=args.diagnostics,
    )
    # Runtime toggles (attached dynamically to keep dataclass stable)
    setattr(cfg, 'enable_jacobi_pc', not args.no_preconditioner)
    setattr(cfg, 'assert_invariants', args.assert_invariants)
    if args.seed is not None:
        setattr(cfg, 'seed', int(args.seed))

    # Validate config early
    validate_config(cfg)
    freeze_config(cfg)
    tracker = ResidualManager()
    expected_hash = None
    expected_core = None
    if args.restart:
        try:
            from .io.checkpoint import load_checkpoint
            state, meta = load_checkpoint(args.restart)
            start_it = int(meta.get('iteration', 0)) + 1
            expected_hash = meta.get('config_hash')
            expected_core = meta.get('core_config')
        except Exception as e:
            print(f"Failed to load checkpoint {args.restart}: {e}")
            state = allocate_state(cfg.nx, cfg.ny)
            start_it = 0
    else:
        state = allocate_state(cfg.nx, cfg.ny)
        start_it = 0
    driver = SimulationDriver(cfg, state, tracker)

    # Restart hash enforcement
    if expected_hash is not None:
        live_hash = config_hash(cfg)
        if live_hash != expected_hash:
            live_core = config_core_dict(cfg)
            print(f"ERROR: Configuration mismatch with checkpoint.\n  checkpoint hash: {expected_hash}\n  current    hash: {live_hash}")
            if isinstance(expected_core, dict):
                added = sorted(k for k in live_core.keys() if k not in expected_core)
                removed = sorted(k for k in expected_core.keys() if k not in live_core)
                changed = sorted(k for k in live_core.keys() if k in expected_core and live_core[k] != expected_core[k])
                if added:
                    print("  Added keys:", ", ".join(added))
                if removed:
                    print("  Removed keys:", ", ".join(removed))
                if changed:
                    print("  Changed:")
                    for k in changed:
                        print(f"    {k}: checkpoint={expected_core[k]!r} current={live_core[k]!r}")
            print("Aborting run. Adjust parameters or use matching checkpoint config.")
            return 3

    json_mode = args.json_stream
    # Force quiet internal diagnostics when streaming JSON to guarantee clean machine output
    if json_mode:
        setattr(cfg, 'force_quiet', True)
    continuity_threshold = args.continuity_threshold
    max_steps = args.steps

    if args.log_jsonl:
        setattr(cfg, 'log_path', args.log_jsonl)
    # Seed control
    if getattr(cfg, 'seed', None) is not None:
        import numpy as _np
        _np.random.seed(int(getattr(cfg, 'seed')))
    last_diag = None; last_state = None; last_residuals = None
    try:
        for st, residuals, diag in driver.run(max_steps=max_steps,
                                             start_iteration=start_it,
                                             progress=(args.progress and not json_mode),
                                             checkpoint_path=args.checkpoint,
                                             checkpoint_interval=args.checkpoint_interval):
            last_diag = diag; last_state = st; last_residuals = residuals
            if json_mode:
                payload = {
                    "type": "step",
                    "iteration": diag.get("iteration"),
                    "dt": diag.get("dt"),
                    "CFL": diag.get("CFL"),
                    "continuity": residuals.get("continuity"),
                    "wall_time": diag.get("wall_time"),
                    "residuals": {
                        'Ru': residuals.get('Ru'),
                        'Rv': residuals.get('Rv'),
                        'Rp': residuals.get('Rp'),
                        'continuity': residuals.get('continuity')
                    },
                    "diagnostics": {k: v for k, v in diag.items() if k not in ("iteration", "dt", "CFL", "wall_time")},
                }
                if getattr(cfg, 'seed', None) is not None:
                    payload.setdefault('run_meta', {})
                    # Access via getattr to appease static type checkers (seed injected dynamically)
                    payload['run_meta']['seed'] = getattr(cfg, 'seed')
                sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
                sys.stdout.flush()
            # Early stop condition
            if continuity_threshold is not None and residuals.get("continuity", 1.0) < continuity_threshold:
                if not json_mode:
                    print(f"Stopping early: continuity {residuals['continuity']:.3e} < threshold {continuity_threshold:.3e}")
                break
    except KeyboardInterrupt:
        # Graceful shutdown: attempt final checkpoint
        if not json_mode:
            print("KeyboardInterrupt received – attempting graceful checkpoint...")
        ck_path = args.checkpoint or "interrupt_final_checkpoint.npz"
        if last_state is not None and last_diag is not None:
            try:
                from .io.checkpoint import save_checkpoint
                save_checkpoint(ck_path, last_state, last_diag.get('iteration', 0), last_diag.get('wall_time', 0.0), cfg)
                if not json_mode:
                    print(f"Final checkpoint written: {ck_path}")
            except Exception as e:  # pragma: no cover
                if not json_mode:
                    print(f"Failed to write final checkpoint: {e}")
        return 130  # POSIX signal 2 convention
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
