# Project Status Dossier

Generated: 2025-08-30

## 1. File Tree (relative paths)
See `file_tree.txt` in this folder (12,460 file entries including virtual env). Only a truncated head shown here:
```
$(head -n 120 file_tree.txt)
```

## 1a. Full Directory Tree
Complete tree (including virtual environment) stored in `tree_full.txt` (14,765 lines). For external reviewer: exclude `.venv/` for logical project size.

## 2. Dependency Manifests
### requirements.txt
```
numpy
scipy
```
### dev-requirements.txt
```
pytest
pytest-cov
ruff
mypy
coverage
jsonschema
radon
```
### optional-requirements.txt
```
matplotlib
numba
```
### pyproject.toml (tooling configs)
```
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "B"]
ignore = []

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

## 3. Sample Simulation Configuration (`cases/lid_cavity_re100.json`)
Current (updated for physical benchmark):
```
{
  "schema_version": 1,
  "nx": 33,
  "ny": 33,
  "lx": 1.0,
  "ly": 1.0,
  "Re": 100.0,
  "solver": "physical",
  "max_iter": 800,
  "tol": 5e-5,
  "cfl_target": 0.5,
  "cfl_growth": 1.1,
  "lin_tol": 1e-10,
  "lin_maxiter": 400,
  "lid_velocity": 1.0,
  "disable_advection": false,
  "test_mode": false,
  "keep_lid_corners": true
}
```

## 4. Test Suite Health
Latest stabilized (post Phase A): All existing core tests pass after physical Poisson projection integration. Added benchmark test pending first validated pass (see Section 11). Legacy instability test now green. Warnings remain (pytest unknown mark `perf`, datetime.utcnow deprecation).

## 5. Key Risk / Focus Areas (Updated)
- Core divergence instability mitigated: physical Poisson projection in place.
- Remaining technical debt: pressure solver diagnostics (iterations/residual), perf mark registration, datetime deprecation, potential need for upwind advection to match benchmark accuracy at higher Re.
- Benchmark risk: current central advection + coarse grid may not meet tight Ghia tolerances; tolerance relaxed initially (RTOL=0.12) while physics matured.

## 6. Completed Phase A Remediation
Implemented:
1. Heuristic projection removed; physical Poisson solve with CG added (`pressure_solver.py`).
2. Explicit operator signatures (divergence, gradient, laplacian) with dx, dy enforced.
3. Legacy projection loops deleted; multi-step divergence test now passes.
4. Operator unit tests added (`test_operators.py`).
5. Backward compatibility alias `P_cache` retained.

## 7. Metrics To Add (Still Pending)
- Pressure solver iterations & residual exposure.
- Divergence history series in tracker.
- Max cell divergence diagnostic.

## 8. Tooling / Quality
- Ruff + mypy configs present; enable CI gating after benchmark validation.
- Pending: register `perf` mark, update datetime usage.
- Consider adding simple timing harness for projection step.

## 9. Open Data Artifacts
- `test_run_results.json` (recent run history)
- `project_status/full_test_output.txt`
- `benchmarks/ghia_centerline_u_re100.json`

## 10. Roadmap (Updated)
Phase A (Stabilization): COMPLETE.

Phase B (Benchmark Validation - In Progress):
  - Validate lid-driven cavity vs Ghia Re=100 centerline (current test added).
  - Tighten tolerance iteratively (target RTOL <= 0.07 after physics tuning).
  - Add Re=400 & 1000 profiles once Re=100 passes stable.

Phase C (Performance & Quality):
  - Preconditioning (Jacobi / ILU) for Poisson.
  - Solver diagnostics & timing instrumentation.

Phase D (Documentation & UX):
  - README algorithm & benchmark results table.
  - Auto-generated config schema docs.

Phase E (Extended Physics - Future):
  - Upwind / QUICK advection schemes.
  - Variable density & transient source terms.

## 11. Benchmark Progress Log
| Timestamp | Event | Detail |
| --------- | ----- | ------ |
| 2025-08-30 | Added reference data | `benchmarks/ghia_centerline_u_re100.json` created |
| 2025-08-30 | Case upgraded | `cases/lid_cavity_re100.json` to physical solver 33x33 |
| 2025-08-30 | Runner extended | Added `y_centerline` capture in `run_case.py` |
| 2025-08-30 | Test added | `test_benchmark_lid_cavity_re100.py` (RTOL=0.12) |
| 2025-08-30 | Schema updated | Added new boolean flags to `config_schema.json` |

Pending entry: first full-suite run including benchmark (will snapshot output to `post_benchmark_tests.txt`).

---
End of dossier.
