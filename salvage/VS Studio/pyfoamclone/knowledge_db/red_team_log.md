# Red Team Log

## 2025-08-30 Snapshot

### Scope
State dataclass integration, residual system unification, synthetic solver scaffolding, test additions, packaging setup (setuptools), convergence gate.

### Findings
1. Dual residual systems linger (numerics/residuals.py still present) causing confusion.
2. Solver still contains synthetic momentum/pressure operators (identity-like) giving false stability impression.
3. Synthetic injection path in run_case un-gated in production flow (risk of silent data fabrication).
4. Poisson convergence test not enforced in CI gating yet (only local).
5. Lack of deterministic seed / reproducibility controls for synthetic profiles.
6. Incomplete error handling in run_loop (partial step state may be left inconsistent on failure; no rollback flag).
7. Packaging metadata placeholder (author, description disclaimers good but no license file verification step).
8. No performance regression guard after refactor (timing harness exists but not asserted in tests).
9. Residual drop metric changed (orders calc) but no unit test directly validating the diagnostic value.
10. Terminal test execution issues hide potential failures (needs script or task integration).
11. allocate_state default fields fixed (u,v,p); custom fields test added but no negative test for missing required fields.
12. Missing type coverage for dynamic config properties (cfl_target etc.)—should enforce via Config dataclass.
13. Pressure correction placeholder may mask divergence; no continuity residual assertion threshold test.
14. No logging level control exposed via config/environment (verbosity not adjustable).
15. absence of docstring for run_case synthetic injection risk warning.

### Risk Ranking (High/Med/Low)
H: (1) Dual residual modules, (2) Synthetic operators, (3) Injection path, (4) Missing CI gate, (13) Pressure placeholder.
M: (5) Reproducibility, (6) Incomplete failure atomicity, (9) Residual diagnostic untested, (10) Test harness fragility, (12) Config typing gaps.
L: (7) Packaging polish, (8) Perf guard, (11) Negative tests, (14) Logging verbosity, (15) Missing warning docs.

### Immediate Remediations Planned
- Remove numerics/residuals.py; migrate any lingering imports.
- Add env flag HARD_FAIL_SYNTHETIC to abort if synthetic operators used outside tests; mark run_case injection with explicit warning log.
- Implement simple 5-point diffusion stencil for momentum to replace identity to surface stability/residual realism.
- Add pytest marker @pytest.mark.convergence_gate and fail if order <1.8; integrate in CI description.
- Add seed control: environment variable PYFOAMCLONE_SEED; set numpy.random.seed when present.
- Add test for residual_drop_Ru_orders diagnostic numeric behavior.

### Deferred (Next Cycle)
- Real pressure Poisson solve coupling with velocity divergence.
- Performance baseline assertion (max total runtime for small grid).
- Config dataclass expansion & validation reinforcement.
- Verbosity and structured logging level control.

---
## Action Items Checklist (Open)
- [ ] Remove numerics/residuals.py and update imports.
- [ ] Add warning & guard for synthetic injection.
- [ ] Implement basic diffusion stencil in momentum_assembly.
- [ ] Add convergence gate marker enforcement.
- [ ] Seed control implementation.
- [ ] Residual drop diagnostic test.

## History
2025-08-30: Initial log created consolidating prior red-team observations; prioritized remediation list added.
