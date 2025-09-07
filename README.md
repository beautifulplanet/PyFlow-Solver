# PyFlow CFD – v1.0.0

High‑clarity, educational incompressible CFD projection solver featuring a reusable simulation driver, structured CLI with JSON streaming, live Plotly dashboard, natural‑language AI control layer, robustness (checkpoint/restart, NaN emergency snapshot, structured logging), and a preconditioned pressure solve. Designed as a portfolio‑grade demonstrator: easy to run, inspect, and extend.

## Features

### Core Solver
* 2D lid‑driven cavity (uniform Cartesian grid) using projection method
* Coherent gradient/divergence pair ⇒ analytically consistent Laplacian
* Matrix‑free Conjugate Gradient pressure solve with optional Jacobi preconditioner
* Adaptive dt safeguard (divergence trend based) & CFL target ramp

### Simulation Driver
* Generator pattern (`SimulationDriver.run`) yields `(state, residuals, diagnostics)` per step
* Centerline velocity profile extraction for visualization

### CLI (`pyflow cli`)
Subcommand architecture (Phase 2):
* `pyflow run` – run a simulation.
* `pyflow validate` – print normalized config + `config_hash`.
* `pyflow show-config` – brief summary + hash.
Core arguments: grid size, Reynolds number, lid velocity, advection scheme, linear solver tolerances, checkpoint / restart controls.
Key flags:
* `--json-stream` emits compact JSON lines (stable schema)
* `--checkpoint`, `--checkpoint-interval N` (0 disables), `--restart CK.npz`
* `--allow-hash-mismatch` (override restart safety)
* `--log-jsonl PATH` structured per-step logging
* `--progress` simple progress bar (suppressed in json stream)
* `--continuity-threshold X` early stop
Config hash enforcement:
* Each checkpoint stores `structured_config_hash`.
* On restart mismatch: exit code 3 + refusal message.
* Override only when intentional using `--allow-hash-mismatch`.
See `CONFIGURATION.md` for full field definitions and hashing details.

### Live Dashboard (`pyflow.dashboard.live_dashboard`)
* Consumes CLI JSON stream via subprocess
* Real‑time residuals, continuity, dt, CFL, centerline velocity profile
* Pause / resume / stop controls, log vs linear scaling, adjustable update interval

### AI Control Layer
* Natural language → structured config parser (e.g. “run 200 steps at Re 500 on a 96x64 grid”)
* Launches CLI as a subprocess; streams results programmatically

### Robustness & Forensics
* Periodic checkpoint & restart (bit‑for‑bit verified by tests)
* Emergency checkpoint on NaN/Inf detection (`*_FAIL_NAN.npz`)
* Structured JSONL logging (per‑step + error events)
* Test coverage for preconditioner correctness, restart integrity, NaN detection

### Benchmark Harness (Frozen for v1.0 Demo)
* Included but de‑prioritized: `pyflow.benchmark.harness` (baseline measurement tool)

## Quickstart (One Command Demo)

Install (editable):
```
pip install -e .[dash,dev]
```

Run the end‑to‑end demo (short cavity simulation with dashboard + logging + checkpoint):
```
python demo.py
```

Headless run (JSON streaming 50 steps on 64x64 grid):
```
pyflow run --nx 64 --ny 64 --steps 50 --json-stream --checkpoint ck.npz --checkpoint-interval 25
```

Resume from checkpoint for 20 more steps:
```
pyflow run --nx 64 --ny 64 --steps 20 --restart ck.npz --json-stream
```

Validate configuration & show hash:
```
pyflow validate --nx 32 --ny 32 --re 250
```

Launch live dashboard (optional `dash`, `plotly` installed):
```
python -m pyflow.dashboard.live_dashboard --nx 64 --ny 64 --steps 200
```

AI natural language control is accessible via the internal parser module (programmatic usage pattern; not exposed as standalone CLI for v1.0).

## Demo Script (What It Does)
`demo.py` performs a short run with:
* Medium grid (64×64)
* JSON logging & structured log file
* Automatic checkpointing
* Prints summary lines and exit residual

## JSON Stream Schema (Stable v1)
Each line: `{"type":"step", "iteration": int, "dt": float, "CFL": float, "continuity": float, "wall_time": float, "residuals": {...}, "diagnostics": {...}}`

Key residuals: `Ru, Rv, Rp, continuity`. Diagnostics include solver iterations (`Rp_iterations`), linear residual norm, divergence norms, centerline velocity profile, checkpoint flags.

## Development & Tests
Run tests:
```
pytest -q
```

## Roadmap (v2.0 Backlog – Not in v1.0)
* Performance profiling & scaling benchmarks integration
* Advanced preconditioners (ILU, multigrid), JIT (Numba) acceleration
* Non‑uniform / generalized mesh abstraction completion
* Extended AI interface (adaptive control policies)

## License
MIT

## Attribution / Purpose
Portfolio‑ready educational CFD codebase emphasizing clarity, observability, and reliability over premature optimization in v1.0.

