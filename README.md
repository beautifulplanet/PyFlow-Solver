
# CFD Solver

A reference Python CFD solver with test-driven development, robust CI, and full boundary condition and logic validation.

## Features
- Pressure projection and divergence-free velocity enforcement
- Lid-driven cavity and manufactured solution tests
- Modular, extensible codebase
- Automated CI with linting, type checks, and tests

## Installation
```sh
pip install -r requirements.txt
```

## Running Tests
```sh
pytest
```

## Directory Structure
- `cfd_solver_new/src/pyflow/` — Core solver code
- `cfd_solver_new/tests/` — Test suite

## Usage Example
See test files in `cfd_solver_new/tests/` for usage patterns and solver configuration.

## Command-Line Interface (Phase 2)

PyFlow now ships with a structured command-line interface wrapping the reusable `SimulationDriver`. It supports human-readable progress output as well as machine-readable JSON line streaming for dashboards or downstream automation.

### Basic Run

```powershell
python -m pyflow.cli --nx 64 --ny 64 --steps 50 --re 100
```

### Key Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--nx` | Interior grid cells in x | 64 |
| `--ny` | Interior grid cells in y | 64 |
| `--re` | Reynolds number | 100.0 |
| `--lid-velocity` | Lid (top boundary) velocity | 1.0 |
| `--steps` / `--max-steps` | Maximum number of time steps | 100 |
| `--scheme {quick,upwind}` | Advection scheme | quick |
| `--disable-advection` | Disable advection predictor stage | off |
| `--cfl` | Target CFL number | 0.5 |
| `--cfl-growth` | CFL growth limiter factor | 1.05 |
| `--lin-tol` | Linear solver tolerance | 1e-10 |
| `--lin-maxiter` | Linear solver max CG iterations | 300 |
| `--json-stream` | Emit one compact JSON object per step | off |
| `--continuity-threshold` | Early stop if continuity residual below value | None |
| `--progress` | Print compact per-step progress (disabled if `--json-stream`) | off |
| `--diagnostics` | Verbose internal solver diagnostics (disabled if `--json-stream`) | off |

Notes:
* When `--json-stream` is active, internal diagnostics are forcibly silenced to guarantee clean machine-readable output (a single JSON object per line).
* The `--progress` flag is ignored in JSON streaming mode.

### JSON Streaming Mode

Produce structured telemetry for each step:

```powershell
python -m pyflow.cli --nx 32 --ny 32 --steps 3 --json-stream
```

Example output (one line per step):

```json
{"type":"step","iteration":0,"dt":0.01,"CFL":0.42,"continuity":3.1e-02,"wall_time":0.012,"residuals":{"Ru":0.12,"Rv":0.11,"Rp":0.09,"continuity":3.1e-02},"diagnostics":{"Rp_iterations":7,"Rp_residual":1.4e-10,"divergence_norm":2.2e-03}}
```

Field schema (stable draft v1):

* `type`: Always `"step"` (reserved for future event types).
* `iteration`: Zero-based step counter.
* `dt`: Time step size actually used.
* `CFL`: Estimated Courant–Friedrichs–Lewy number after the step.
* `continuity`: Continuity (divergence) residual after projection.
* `wall_time`: Seconds elapsed since simulation start.
* `residuals`: Core solver residual bundle (may expand in later versions).
* `diagnostics`: Extended pressure / linear solve diagnostics (content may evolve additively; keys are reserved for quantitative values only).

Backward compatibility policy: New keys may be added; existing keys will not be removed or change semantic meaning within the Phase 2 contract. Consumers should ignore unknown keys.

### Early Stopping by Continuity

Stop automatically when continuity residual is sufficiently small:

```powershell
python -m pyflow.cli --nx 64 --ny 64 --steps 500 --continuity-threshold 1e-5 --progress
```

### Verbose Diagnostics (Human Mode)

To inspect the pressure projection internals (matrix-free CG iteration diagnostics), enable diagnostics without JSON streaming:

```powershell
python -m pyflow.cli --nx 32 --ny 32 --steps 5 --diagnostics
```

### Integration Pattern

For programmatic consumption, treat each JSON line as an independent object:

```python
import json, subprocess
proc = subprocess.Popen([
	"python", "-m", "pyflow.cli", "--nx", "32", "--ny", "32", "--steps", "10", "--json-stream"],
	stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
	obj = json.loads(line)
	# feed to dashboard or adaptive controller
```

### Testing

The test suite contains `test_cli_json_stream.py` which asserts that JSON streaming mode produces exactly the expected number of clean lines (no stray prints). This guards the machine contract.

### Roadmap (Post Phase 2)

Planned but not yet implemented features include:
* Snapshot / restart checkpointing
* Adaptive dt controllers exposed via CLI flags
* Event hooks and plugin callbacks
* Structured error / warning events in the JSON stream

---
Phase 2 (CLI & Streaming) deliverables are now complete and validated.

## Live Dashboard (Phase 3 – In Progress)

A Plotly Dash application can visualize a running simulation in real time. It launches the CLI in a subprocess with `--json-stream` and renders:

* Residual history (`Ru`, `Rv`, `Rp` if present)
* Continuity (mean-free divergence) residual
* Horizontal velocity profile along the vertical centerline (future extension – placeholder hooks included)

### Install Optional Dependencies

```powershell
pip install dash plotly
```

### Launch

```powershell
python -m pyflow.dashboard.live_dashboard --nx 64 --ny 64 --steps 500 --re 100
```

Open http://127.0.0.1:8050 in a browser. Use `--port` to change the port. Additional CLI flags for the underlying simulation can be appended after `--extra-args`:

```powershell
python -m pyflow.dashboard.live_dashboard --nx 128 --ny 128 --steps 800 --extra-args --cfl 0.4 --scheme upwind
```

### Notes
* The dashboard reads only the stable JSON line protocol (decoupled from internals).
* If Dash/Plotly are not installed, a clear message is displayed.
* Centerline velocity profile will populate once the solver begins emitting `u_centerline` in diagnostics (future task).

---

## License
See `LICENSE` file.
