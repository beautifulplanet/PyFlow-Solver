# PyFlow Configuration

This document describes the user‑facing simulation configuration model (`SimulationConfig`) and related CLI usage.

## Overview
Configuration is normalized and validated via Pydantic. A deterministic 12‑char SHA1 hash (`config_hash`) is computed over the normalized payload (excluding ephemeral runtime fields) and embedded into checkpoints. On restart, the stored hash must match the current config unless `--allow-hash-mismatch` is supplied.

## Fields
| Field | Type | Default | Constraints | Notes |
|-------|------|---------|-------------|-------|
| nx | int | 64 (CLI) | >=3 | Grid cells in x (interior) |
| ny | int | 64 (CLI) | >=3 | Grid cells in y (interior) |
| Re | float | 100.0 | >0 | Reynolds number |
| lid_velocity | float | 1.0 |  | Top lid velocity |
| cfl_target | float | 0.5 | 0< val <=1 | Target CFL |
| cfl_growth | float | 1.05 | 1< val <=1.2 | Adaptive CFL growth factor |
| lin_tol | float | 1e-10 | 1e-14 <= val <= 1e-2 | Linear solver tolerance |
| lin_maxiter | int | 200 (config) / 300 (CLI) | 1..10000 | Max linear solver iterations |
| advection_scheme | enum | upwind | {'upwind','quick'} | Ignored if `--disable-advection` |
| disable_advection | bool | false |  | Disables convective term |
| diagnostics | bool | true |  | Enables verbose per‑step diagnostics (suppressed in json stream) |
| log_path | str? | None |  | If set, JSONL structured logs are written |
| checkpoint_interval | int? | None (CLI sentinel 0) | >=1 if provided | Periodic checkpointing frequency |
| emergency_checkpoint_path | str? | None |  | Reserved for interruption handling |
| seed | int? | None | 0 <= val < 2^32 | RNG seeding for reproducibility |
| turbulence_model | str? | None |  | Reserved future feature |
| schema_version | int | 1 |  | Internal version marker |

### Sentinel Behavior
`--checkpoint-interval 0` (default) maps to `None` meaning disabled. Any positive integer enables periodic checkpoint saves.

## Hash Semantics
Hash includes all model fields except runtime‑only / ephemeral values (`log_stream`). Ordering is normalized via JSON canonicalization (sorted keys, compact separators). Changing any hashed field changes the restart hash and will block mismatched restarts unless overridden.

## CLI Commands
`pyflow run` – Execute a simulation.
`pyflow validate` – Print normalized config + hash as JSON.
`pyflow show-config` – Show brief summary + hash.

### Common Arguments
```
--nx --ny --re --lid-velocity --scheme {quick,upwind} --disable-advection \
--cfl --cfl-growth --lin-tol --lin-maxiter --checkpoint-interval 0 \
--log-jsonl PATH --seed SEED --diagnostics
```

### run‑specific
```
--steps N --checkpoint PATH --restart CKPATH --allow-hash-mismatch \
--json-stream --continuity-threshold VAL --progress --assert-invariants
```

### Restart Safety
On `--restart ck.npz` the loader reads `structured_config_hash` from the checkpoint metadata and compares to current `config_hash`.
- Match: proceed.
- Mismatch: exit code 3 with explanatory message.
- Override: supply `--allow-hash-mismatch` (use only if you knowingly changed parameters and accept non‑reproducibility).

## Programmatic Usage
```python
from pyflow.config.model import SimulationConfig
cfg = SimulationConfig(nx=64, ny=64, Re=200.0)
print(cfg.config_hash)
```

## Soft Warnings
Extreme aspect ratios (nx/ny > 10 or < 0.1) set an internal `_soft_warning` attribute (future: surface via `validate`).

## Future Extensions
- Turbulence closure selection
- Mesh / geometry descriptors
- Structured emission of soft warnings in CLI output
- Versioned schema migrations

## Exit Codes (Selected)
| Code | Meaning |
|------|---------|
| 0 | Success / normal completion |
| 2 | Configuration validation error |
| 3 | Restart refused due to config hash mismatch |
| 130 | Keyboard interrupt with graceful checkpoint |

## Quick Examples
Validate config:
```
pyflow validate --nx 32 --ny 32 --re 250
```
Run with periodic checkpoints every 50 steps:
```
pyflow run --nx 64 --ny 64 --checkpoint-interval 50 --steps 500 --checkpoint run.ck
```
Attempt restart (expected to match hash):
```
pyflow run --restart run.ck --steps 100
```
Force restart despite changed params:
```
pyflow run --restart run.ck --nx 80 --allow-hash-mismatch --steps 50
```
