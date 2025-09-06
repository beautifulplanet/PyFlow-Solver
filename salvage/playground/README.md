# CFD Playground (v0.1)

Purpose: Isolated experimental scripts to prototype, validate, and de-risk concepts prior to production solver implementation. Each script is self-contained, minimal-dependency, and emits JSONL artifacts for analysis.

Run Conventions:
- All scripts accept `--output-dir` (default: `./artifacts` inside playground) and create it if missing.
- Randomness: deterministic seeds (if used) logged in first line.
- Telemetry: JSON Lines with a leading schema version field.

Scripts (Initial Set):
- `manufactured_solution_order.py` – Verifies spatial order for gradient/Laplacian prototypes.
- `adaptive_dt_pid_sim.py` – Simulates PID timestep adaptation dynamics.
- `residual_plateau_classifier_demo.py` – Demonstrates plateau detection robustness.
- `stencil_symmetry_audit.py` – Audits matrix symmetry & boundary coefficient patterns.
- `config_schema_validator.py` – Validates sample configs against draft schema.
- `duplication_similarity_demo.py` – Shows TF-IDF similarity and threshold tuning dataset.
- `microbench_kernels.py` – Benchmarks primitive operations & normalizes metrics.

Artifacts Directory Structure:
```
playground/
  artifacts/
    manufactured/
    adaptive_dt/
    plateau/
    stencil/
    config/
    duplication/
    microbench/
```

Extension Roadmap:
- Embedding similarity demo
- Multigrid prototype timing
- Mixed precision pressure solve sensitivity
- Failure injection harness preview

Usage Example:
```
python playground/manufactured_solution_order.py --grids 32 64 128 256 --output-dir playground/artifacts/manufactured
```

Analysis Suggestions:
- Plot log(err) vs log(h) slope for order.
- Examine adaptive dt variance and settling time.
- Evaluate false positive rate of plateau classifier under noise injection.

(End README v0.1)
