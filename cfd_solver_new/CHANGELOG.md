# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and adheres to Semantic Versioning. Because `v1.0.0` is the first public/portfolio release, prior internal iterations are summarized historically rather than enumerated as individual tagged versions.

## [1.0.0] - 2025-09-06
### Summary
First stable, feature‑complete release of the PyFlow CFD educational / portfolio solver. Represents the transformation of an initial brittle prototype into a coherent, test‑hardened, demonstrable product with a full usability and observability stack.

### Added
* Coherent finite‑difference / finite‑volume hybrid operators (gradient, divergence, Laplacian) ensuring `laplacian ≈ div(grad)` consistency to roundoff.
* Projection method core with matrix‑free Conjugate Gradient pressure solve and optional Jacobi preconditioning.
* Reusable generator‑based `SimulationDriver` yielding per‑step `(state, residuals, diagnostics)`.
* Comprehensive CLI (`pyflow.cli`) with:
  - Grid, Reynolds number, advection scheme controls.
  - JSON line streaming (`--json-stream`) for machine ingestion.
  - Early stop threshold, progress mode.
  - Checkpointing (`--checkpoint`, `--checkpoint-interval`) and restart (`--restart`).
  - Structured logging (`--log-jsonl`).
  - Preconditioner toggle (`--no-preconditioner`).
* Live Plotly Dash dashboard (residuals, continuity, dt, CFL, centerline profile, pause/resume/stop, scaling toggle, adjustable interval).
* AI Control Layer: natural language → structured configuration → launched simulation.
* Robustness features:
  - Periodic checkpoint + verified restart integrity.
  - Emergency checkpoint on NaN/Inf detection with `_FAIL_NAN` suffix.
  - Structured JSONL log entries (steps + error events).
  - NaN/Inf detection integrated into solver step diagnostics.
* Test suite expansions:
  - Operator coherence & projection contract tests.
  - Driver iteration behavior.
  - CLI JSON streaming integrity.
  - AI parsing + launch.
  - Checkpoint / restart bit‑for‑bit equivalence.
  - Preconditioner correctness (solution equality) and iteration improvement.
  - NaN detection & logging validation.
  - Benchmark harness smoke test (harness itself frozen for post‑1.0 roadmap).
* Benchmark harness (de‑prioritized for 1.0 usage) to measure scaling (retained but not part of release objectives).
* `demo.py` quickstart script for one‑command portfolio demonstration.
* `pyproject.toml` with pinned core dependencies, optional extras, and dev tooling.

### Changed
* Refactored initial ad‑hoc pressure solver path into a clean, matrix‑free operator + optional sparse assembly for preconditioning.
* Consolidated verbose diagnostics control (`diagnostics` flag + `force_quiet` for clean JSON mode).
* README completely rewritten to reflect final feature set and usage patterns.

### Removed
* Legacy / flaky regression and performance tests superseded by robust new test suite (removed `tests/legacy/`).

### Security / Stability
* All new critical features (checkpointing, preconditioner, NaN detection, logging) receive explicit unit tests to prevent silent regressions.
* Emergency snapshot ensures forensic capture on catastrophic numerical failure.

### Known Limitations (Deferred to v2.0 Backlog)
* Uniform Cartesian grid only (no stretched / unstructured meshes yet).
* No advanced preconditioners (ILU, multigrid) or JIT acceleration (Numba) integrated.
* Benchmark harness present but not yet integrated into CI performance gating.
* Single‑threaded CPU execution; no GPU support.

### Roadmap (Post‑1.0 Themes)
1. Performance Profiling & Scaling (activate harness, identify true hotspots).
2. Advanced Linear Solvers / Preconditioners.
3. Mesh Generalization Layer (cell / face topology abstraction).
4. Optional JIT / vectorization for advection & diffusion paths.
5. Extended AI control (adaptive timestepping / goal‑driven runs).

---

Released under the MIT License. This changelog documents the stabilization and hardening journey culminating in a polished, demonstrable v1.0.0 release.
