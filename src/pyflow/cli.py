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
import argparse, json, sys
from typing import Any
from .drivers.simulation_driver import SimulationDriver
from .core.ghost_fields import allocate_state
from .residuals.manager import ResidualManager
from .config.model import SimulationConfig, ConfigError
from pydantic import ValidationError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PyFlow CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--nx", type=int, default=64)
        sp.add_argument("--ny", type=int, default=64)
        sp.add_argument("--re", type=float, default=100.0)
        sp.add_argument("--lid-velocity", type=float, default=1.0)
        sp.add_argument("--scheme", choices=["quick","upwind"], default="quick")
        sp.add_argument("--disable-advection", action="store_true")
        sp.add_argument("--cfl", type=float, default=0.5)
        sp.add_argument("--cfl-growth", type=float, default=1.05)
        sp.add_argument("--lin-tol", type=float, default=1e-10)
        sp.add_argument("--lin-maxiter", type=int, default=300)
        # 0 means: no periodic checkpoints (mapped to None for model which requires >=1)
        sp.add_argument("--checkpoint-interval", type=int, default=0)
        sp.add_argument("--log-jsonl", type=str, default=None)
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--diagnostics", action="store_true")

    run_p = sub.add_parser("run", help="Run a simulation")
    add_common(run_p)
    run_p.add_argument("--steps", type=int, default=100)
    run_p.add_argument("--checkpoint", type=str, default=None)
    run_p.add_argument("--restart", type=str, default=None)
    run_p.add_argument("--allow-hash-mismatch", action="store_true")
    run_p.add_argument("--json-stream", action="store_true")
    run_p.add_argument("--continuity-threshold", type=float, default=None)
    run_p.add_argument("--progress", action="store_true")
    run_p.add_argument("--assert-invariants", action="store_true")

    val_p = sub.add_parser("validate", help="Validate and display normalized config")
    add_common(val_p)

    show_p = sub.add_parser("show-config", help="Show normalized config (hash only)")
    add_common(show_p)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Map sentinel 0 -> None before constructing config
        _ck_int = None
        if hasattr(args, 'checkpoint_interval'):
            ci = getattr(args, 'checkpoint_interval')
            if ci not in (None, 0):
                _ck_int = ci
        sim_cfg = SimulationConfig(
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
            checkpoint_interval=_ck_int,
            log_path=args.log_jsonl,
            seed=args.seed,
            turbulence_model=None,
        )
    except (ValidationError, ConfigError) as e:
        print("Configuration invalid:\n" + str(e))
        return 2

    if args.command == 'validate':
        out = sim_cfg.model_dump()
        out['config_hash'] = sim_cfg.config_hash
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    if args.command == 'show-config':
        print(json.dumps({'hash': sim_cfg.config_hash, 'brief': sim_cfg.brief()}, indent=2))
        return 0

    tracker = ResidualManager()
    if args.command == 'run' and args.restart:
        try:
            from .io.checkpoint import load_checkpoint
            state, meta = load_checkpoint(args.restart)
            ck_hash = meta.get('structured_config_hash')
            if ck_hash and ck_hash != sim_cfg.config_hash and not args.allow_hash_mismatch:
                print(f"Refusing restart: checkpoint hash {ck_hash} != current {sim_cfg.config_hash} (use --allow-hash-mismatch to override)")
                return 3
            start_it = int(meta.get('iteration', 0)) + 1
        except Exception as e:
            print(f"Failed to load checkpoint {args.restart}: {e}")
            state = allocate_state(sim_cfg.nx, sim_cfg.ny)
            start_it = 0
    else:
        state = allocate_state(sim_cfg.nx, sim_cfg.ny)
        start_it = 0
    driver = SimulationDriver(sim_cfg, state, tracker)

    json_mode = getattr(args, 'json_stream', False)
    if json_mode:
        setattr(sim_cfg, 'force_quiet', True)
    continuity_threshold = getattr(args, 'continuity_threshold', None)
    max_steps = getattr(args, 'steps', 0)

    if sim_cfg.seed is not None:
        import numpy as _np
        _np.random.seed(int(sim_cfg.seed))
    last_diag = None; last_state = None; last_residuals = None
    try:
        # Use sanitized interval (original 0 replaced with None)
        runtime_ck_int = None if getattr(args, 'checkpoint_interval', None) in (None, 0) else getattr(args, 'checkpoint_interval')
        for st, residuals, diag in driver.run(max_steps=max_steps,
                                              start_iteration=start_it,
                                              progress=(args.progress and not json_mode),
                                              checkpoint_path=args.checkpoint,
                                              checkpoint_interval=runtime_ck_int):
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
                if sim_cfg.seed is not None:
                    payload.setdefault('run_meta', {})
                    payload['run_meta']['seed'] = sim_cfg.seed
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
        ck_path = getattr(args, 'checkpoint', None) or "interrupt_final_checkpoint.npz"
        if last_state is not None and last_diag is not None:
            try:
                from .io.checkpoint import save_checkpoint
                save_checkpoint(ck_path, last_state, last_diag.get('iteration', 0), last_diag.get('wall_time', 0.0), sim_cfg)
                if not json_mode:
                    print(f"Final checkpoint written: {ck_path}")
            except Exception as e:  # pragma: no cover
                if not json_mode:
                    print(f"Failed to write final checkpoint: {e}")
        return 130  # POSIX signal 2 convention
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
