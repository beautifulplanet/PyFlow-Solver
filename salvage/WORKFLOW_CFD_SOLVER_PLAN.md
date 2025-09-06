# CFD Solver Development Workflow Plan

> Purpose: Operational playbook to build the next-generation CFD solver leveraging prior postmortem insights, salvaged high-quality functions, knowledge DB, semantic index, and governance guardrails. This is an actionable, phase‑gated workflow with embedded experiment & data collection matrix.

## 0. Context Snapshot
- Historical Pain Points: numerical divergence (pressure–velocity coupling, CFL spikes), monolithic functions, duplication, undocumented prototype drift, lack of promotion criteria, noisy failure tracking.
- Assets Available: `reusable_funcs.py`, salvage pool (`orphan_salvage.jsonl`), semantic TF‑IDF index, `cfd_knowledge.db`, evolution and failure taxonomies, complexity pre‑commit hook, refactor plan, function quality metrics, dependency graph.
- Strategic Goals:
  1. Rapid creation of a stable, modular incompressible Navier–Stokes solver (baseline 2D) with extensibility toward turbulence & compressible flows.
  2. Data‑driven iteration—continuous instrumentation & promotion gates.
  3. Minimize reintroduction of historical failure modes.

## 1. High-Level Phase Roadmap
| Phase | Name | Core Output | Gate Metrics (must pass) |
| ----- | ---- | ----------- | ------------------------ |
| P0 | Foundation & Scaffolding | Repo skeleton, config schema, logging infra | Lint passes, complexity OK (<12 avg new funcs) |
| P1 | Core Data Structures | Grid, field arrays, BC registry, linear algebra adapters | Unit coverage ≥70% for structures |
| P2 | Discretization Kernel MVP | Spatial operators (grad/div, Laplacian), time integrator skeleton | Operator verification (manufactured solution err <5%) |
| P3 | Pressure–Velocity Coupling | SIMPLE / PISO loop + Rhie–Chow interpolation | Residual monotonic drop 3 orders for lid cavity |
| P4 | Stabilization & Robustness | Adaptive dt, under‑relax, residual diagnostics | No divergence in 5 canonical cases |
| P5 | Performance & Profiling | Hotspot timing, memory footprint baseline | 2× speed vs naive baseline, memory within budget |
| P6 | Extended Physics & Turbulence Hooks | Turbulence model interface, scalar transport | Plug-in API stable, regression green |
| P7 | Validation Campaign | Benchmark suite (cavity, channel, TGV, shock tube if compressible branch) | All benchmark errors within targets |
| P8 | Packaging & Promotion | Versioned release, docs site, API spec | Promotion checklist 100% satisfied |
| P9 | Continuous Improvement | Drift monitor, duplication guard, embedding upgrade | Drift < threshold, duplication incidents = 0 |

## 2. Directory Scaffold (Target)
```
cfds/
  core/
    grid.py
    fields.py
    linalg_backend.py
    operators/
      gradient.py
      divergence.py
      laplacian.py
      interpolation.py
    coupling/
      simple.py
      piso.py
      rhie_chow.py
    timestep/
      adaptive_dt.py
    bc/
      registry.py
      apply_bc.py
  physics/
    turbulence/
      k_epsilon.py (future)
    scalars/
      advection_diffusion.py
  solvers/
    cavity2d.py
    channel_flow.py
  io/
    config_loader.py
    writer_vtk.py
  utils/
    residuals.py
    logging_setup.py
    profiling.py
    failure_logging_helper.py (linked or imported)
  reuse/
    reusable_funcs.py (copied from existing asset)
  tests/
    test_grid.py
    test_operators.py
    test_coupling.py
    test_residuals.py
  experiments/
    experiment_matrix.md
    scripts/
      run_cavity_convergence.py
      run_tgv_decay.py
      run_channel_re_tau180.py
      run_shock_tube.py (later)
  docs/
    design_overview.md
    api_reference.md
  scripts/
    build_semantic_index.py (symlink or wrapper)
    export_sqlite_db.py (update path refs)
  .generated_hooks/
  requirements.txt / pyproject.toml
```

## 3. Phase Details & Tasks
### Phase 0: Foundation & Scaffolding
Tasks:
- Create directory skeleton.
- Port `reusable_funcs.py` (only vetted salvage entries) into `reuse/`.
- Add logging setup: structured JSON lines + human console handler.
- Integrate `failure_logging_helper.log_failure` at solver loop boundaries.
- Define unified config schema (YAML/JSON) with validated fields (grid size, CFL target, solver tolerances, relaxation factors).
- Implement simple CLI harness: `python -m cfds.solvers.cavity2d --config configs/cavity_baseline.yml`.
Data Collection:
- Capture initial complexity metrics snapshot.
- Save baseline build time & import dependency graph slice.
Gate:
- Lint & type checks pass; complexity avg new functions <12; no function >60 LOC.

### Phase 1: Core Data Structures
Tasks:
- `Grid` (structured uniform first; include placeholder for stretched coordinates).
- `Field` abstraction (numpy array + metadata: centering, units, ghost layers).
- Boundary condition registry: map (field, face) → strategy object.
- Linear algebra backend adapter (start with SciPy sparse or simple numpy; plan to swap to PETSc later).
Experiments:
- Memory layout micro-bench (row-major vs possible struct-of-arrays variant for future GPUs) – record timing for copy, gradient kernel placeholder.
Data Collection:
- Store benchmark JSON: `benchmarks/phase1_memory.json` (size, time, bandwidth).
Gate:
- Unit coverage ≥70%; field initialization speed stable (< threshold vs baseline).

### Phase 2: Discretization Kernel MVP
Tasks:
- Implement finite volume operators on structured grid (central differencing baseline) from salvage interpolation & gradient utilities.
- Manufactured solution test (e.g., u = sin(pi x) sin(pi y)).
- Time integration skeleton (explicit Euler + placeholder mid-step for future RK2).
Experiments:
- Convergence order study (h halved 3×) -> output CSV `experiments/results/convergence_manufactured.csv`.
- Stability test varying CFL (0.1→1.5) logging first divergence point.
Data Collection:
- Residual history JSON lines per run.
Gate:
- Observed spatial order ~2 (within ±0.2 of expected); stable for CFL ≤0.5.

### Phase 3: Pressure–Velocity Coupling
Tasks:
- SIMPLE loop with pressure correction Poisson solve; integrate Rhie–Chow interpolation (from blueprint recommendations).
- Residual tracker stores: L2(u), L2(v), continuity, pressure correction.
- Under-relaxation parameters in config.
Experiments:
- Lid-driven cavity (Re=100, 1000) compare centerline velocity vs Ghia et al. reference; store error metrics.
- Divergence stress test: artificially large dt to confirm controlled failure logged.
Gate:
- Residuals drop ≥1e3; centerline velocity error <5% at Re=100.

### Phase 4: Stabilization & Robustness
Tasks:
- Adaptive dt controller: adjust dt to keep max Courant <= target (with smoothing gain).
- Residual plateau detector triggers optional refinement suggestions.
- Optional deferred correction or SIMPLEC variant toggled by config.
Experiments:
- Long-run (10k steps) drift test cavity; record conservation of mass.
- Parameter sweep for relaxation factors (grid search) – store stability map heatmap data.
Gate:
- Zero net mass flux within tolerance (|Σ mass imbalance|/total <1e-8); no unhandled exceptions in sweeps.

### Phase 5: Performance & Profiling
Tasks:
- Add lightweight profiling decorator to critical kernels; accumulate to JSON.
- Vectorize hotspot loops; optional numba experiment.
- Build performance baseline (wall time per 100 steps for 128×128 cavity).
Experiments:
- Before/after optimization A/B; memory profiling (peak RSS) vs baseline.
Data Collection:
- `performance_baseline.json`, `performance_optim.json`.
Gate:
- ≥2× improvement over naive; memory growth <10%.

### Phase 6: Extended Physics & Turbulence Hooks
Tasks:
- Abstract transport equation assembly (scalar fields) enabling passive scalar test.
- Turbulence model interface (strategy class) – stub implementations.
- Ensure decoupled modular assembly pipeline (no monolith >200 LOC).
Experiments:
- Passive scalar diffusion analytic decay test.
- Turbulence placeholder (eddy viscosity constant) sanity residual behavior.
Gate:
- Scalar diffusion error <5%; interface stable (no cyclic deps in dependency graph diff).

### Phase 7: Validation Campaign
Tasks:
- Taylor–Green vortex decay (compare kinetic energy curve vs analytical early-time decay).
- Channel flow friction factor vs Blasius correlation (laminar first). 
- (If compressible branch) Sod shock tube Riemann problem.
- Mesh refinement study script (auto generate 32→256 grid runs; compute order).
Data Collection:
- `validation_results.sqlite` (store case, metric, reference, error).
Gate:
- All cases pass error thresholds; convergence rates validated within expected ranges.

### Phase 8: Packaging & Promotion
Tasks:
- Auto-generate API docs from docstrings (pdoc or Sphinx).
- Introduce semantic versioning; tag v0.1.0.
- Publish installable package (local index or PyPI if desired).
- Finalize promotion checklist automation script reading metrics artifacts.
Gate:
- Checklist 100%; zero open severity-1 issues; duplication scan <0.9 similarity conflicts.

### Phase 9: Continuous Improvement
Tasks:
- Drift monitor script: compare latest complexity & duplication vs baseline; logs anomalies.
- Extend semantic index build to incremental mode (hash-based skip).
- Optional embedding model integration (sentence-transformers) with fallback to TF-IDF.
Gate:
- Monthly drift report produced; no regression in key metrics.

## 4. Experiment & Data Collection Matrix
| ID | Case | Purpose | Metrics | Artifact |
|----|------|---------|---------|----------|
| E1 | Manufactured solution | Verify spatial order | L2 error, order | `convergence_manufactured.csv` |
| E2 | CFL sweep | Stability boundary | Divergence step, residual patterns | `cfl_sweep.jsonl` |
| E3 | Lid cavity Re=100/1000 | Coupling accuracy | Centerline velocity error | `cavity_validation.json` |
| E4 | Relaxation sweep | Robustness map | Success/fail grid | `relaxation_heatmap.csv` |
| E5 | Long-run drift | Conservation | Mass imbalance trend | `drift_longrun.jsonl` |
| E6 | Performance baseline | Profiling | Time/step, memory | `performance_baseline.json` |
| E7 | Optimization A/B | Speedup quantification | Speedup factor | `performance_optim.json` |
| E8 | Taylor–Green vortex | Physics validation | Energy decay curve error | `tgv_validation.json` |
| E9 | Channel flow | Shear accuracy | Friction factor error | `channel_validation.json` |
| E10 | Mesh refinement | Order confirmation | Order vs grid | `mesh_refinement.csv` |
| E11 | Passive scalar decay | Scalar transport check | L2 error | `scalar_decay.json` |
| E12 | Shock tube | Compressible validation | Shock position error | `shock_tube.json` |

## 5. Metrics & Telemetry
- Residuals: store each n steps (JSON line) {step, res_u, res_v, res_p, continuity, dt}.
- Performance: per-kernel ms, cumulative time, calls.
- Complexity: nightly snapshot (function LOC, cyclomatic if analyzer integrated later).
- Failure Log: unify via `log_failure(event_type, context, meta)` writing to `failures.jsonl`.
- Validation DB: migrate JSON results into SQLite for querying trend regressions.

## 6. Knowledge Reuse Integration
- Semantic Query Pre-Commit: Optional hook suggests existing function if similarity >0.88 for new >50 LOC addition.
- Reuse Annotation: Each reused salvage snippet gains header comment referencing origin (evolution tag + original path) for traceability.
- Periodic Index Refresh: Run index rebuild after merging >10 changes or weekly (whichever first).

## 7. Risk Register (Focused)
| Risk | Trigger | Impact | Mitigation | Early Signal |
|------|---------|--------|-----------|--------------|
| Divergence under fine grid | High CFL or poor relaxation | Delays & instability | Adaptive dt + auto-relax tuning | Residual oscillations |
| Monolithic creep | Rapid feature addition | Hard refactor later | Size guards + checklist | New function LOC spike |
| Silent accuracy regression | Refactor of operators | Wrong physics | Validation suite CI | Energy decay deviation |
| Performance stagnation | No profiling culture | Uncompetitive runtime | Mandatory baseline run P5 | Kernel time plateau |
| Knowledge drift | New code bypasses salvage | Duplication & bugs | Similarity guard + drift monitor | Duplicate similarity >0.9 |

## 8. Automation Hooks
- `scripts/run_experiment.py --matrix E1,E2` executes batch, aggregates summary table.
- `scripts/promote_release.py` validates all gates → creates version tag.
- `scripts/drift_monitor.py` compares current metrics vs baseline, emits alert lines.

## 9. Promotion Checklist (Operational Script Fields)
JSON Schema Example:
```
{
  "tests_pass": true,
  "residual_drop_orders": 3.4,
  "validation_cases_passed": 9,
  "validation_cases_total": 9,
  "avg_function_complexity": 8.7,
  "max_function_loc": 145,
  "duplication_conflicts": 0,
  "doc_coverage": 0.92,
  "performance_speedup": 2.3
}
```
Gate logic: fail if any threshold not met; produce markdown report for release notes.

## 10. Immediate Next Actions (Execution Sequence)
1. Generate scaffold directories & placeholder files (Phase 0).
2. Copy `reusable_funcs.py` into scaffold; trim to referenced kernels only.
3. Implement logging + failure helper integration.
4. Draft config schema & minimal CLI (cavity baseline driver returning residual trace).
5. Add Phase 0 unit tests; run complexity snapshot.
6. Prepare experiment script templates (E1, E2 skeletons) even if kernels not yet implemented (fail gracefully) so data pipeline is ready.

## 11. Data Collection Enhancements (New) 
- Hash Manifest: store file SHA256 + metric snapshot in `artifacts/manifest.json` enabling incremental rebuild decisions.
- Residual Pattern Classifier (future): simple heuristic categorizing convergence (monotonic, oscillatory, plateau, divergent) appended to each run record.
- Automatic Parameter Tuner: For SIMPLE under-relaxation factors (grid search) storing best stable tuple.

## 12. Embedding Upgrade Path
Phase 9 enhancement: add `build_embeddings_index.py` using sentence-transformers (if allowed). Fallback to TF-IDF when embedding model missing. Duplicate detection uses embedding cosine >0.92 as conflict.

## 13. Quality Guardrail Extensions
- Add optional `max_new_duplicate_similarity` threshold (0.88) in pre-commit.
- Enforce docstring presence for all public (non-underscore) functions.
- Complexity trending graph generator (weekly) -> `reports/complexity_trend.md`.

## 14. Validation Targets (Initial)
| Case | Metric | Target |
|------|--------|--------|
| Lid cavity Re=100 | Centerline u error | <5% |
| Lid cavity Re=1000 | Centerline u error | <8% |
| Taylor–Green (early) | Energy decay RMS error | <5% |
| Channel laminar | Friction factor error | <5% |
| Manufactured solution | Order | 1.8–2.2 |

## 15. KPIs (Project Level)
- Time from feature spec → merged (median) <3 days.
- Regression failure rate per release: 0.
- Duplicate similarity alerts per month: <2.
- Automated experiment coverage: ≥90% (of defined matrix executed weekly).

## 16. Change Management
- Every major refactor: capture pre/post performance/accuracy diff in `refactors/ID/report.md`.
- Auto-label PRs with tags: performance, accuracy, refactor, robustness, infra.

## 17. Open Backlog (Prioritized)
1. Scaffolding automation script
2. Residual plateau classifier
3. Drift monitor script
4. Parameter auto-tuner (relaxation factors)
5. Embedding-based semantic index
6. Mesh adaptation prototype (future research)

## 18. Appendices
### A. Residual JSON Line Example
```
{"step": 400, "res_u": 2.1e-3, "res_v": 1.9e-3, "res_p": 9.5e-4, "continuity": 7.4e-5, "dt": 0.001}
```
### B. Failure Log Example
```
{"timestamp": "2025-09-01T10:22:11Z", "event_type": "divergence", "context": {"case": "cavity", "Re": 1000}, "meta": {"last_dt": 0.004, "CFL_max": 1.8}}
```
### C. Profiling Output Snippet
```
{"kernel": "apply_bc_velocity", "calls": 400, "total_ms": 152.7, "avg_ms": 0.382}
```

---
Update this workflow as phases conclude; record deviations & rationale in an `AMENDMENTS` section below.

## AMENDMENTS
(None yet)
