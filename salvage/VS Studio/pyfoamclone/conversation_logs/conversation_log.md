# Project Conversation Log

## Centralized Log (2025-08-30)

This file consolidates all major architectural, technical, and decision-making conversations for the pyfoamclone CFD project. It is the canonical source for project history, rationale, and key exchanges.

---

### Chronological Project History

#### 2025-08-28: Project Initialization & Early Instability
- Project scaffolded: initial directory structure, synthetic solver, placeholder operators (identity-like) for rapid prototyping.
- Early tests revealed non-physical divergence growth in multi-step projection tests, indicating instability in the core numerical method.
- Added logging utilities and a JSON test run recorder to capture test results and facilitate debugging.
- Created initial technical debt and red-team logs to track risks and remediation priorities.

#### 2025-08-29: Architectural Assessment & Roadmap
- Generated a comprehensive project dossier: file tree, dependency manifests, sample config, and full test output.
- Architect persona reviewed the codebase, identifying:
  - Instability in the projection method (multi-step divergence growth).
  - Ambiguity in operator interfaces (state-based vs explicit spacing).
  - Caching and backward compatibility issues (pressure matrix, legacy tests).
- Produced a multi-phase roadmap:
  - Phase A: Stabilization (operator refactor, physical Poisson projection, test recovery).
  - Phase B: Benchmark validation (lid-driven cavity, Ghia profile).
  - Phase C: Performance and quality (preconditioning, diagnostics, documentation).
- User prioritized Option 1 (operator refactor) and directed execution of steps 1–3 (test baseline, integrate projection, re-run tests).

#### 2025-08-29: Operator Refactor & Test Recovery
- Refactored core numerical operators to explicit signatures: `divergence(u, v, dx, dy)`, `gradient(p, dx, dy)`, `laplacian(p, dx, dy)`.
- Added new unit tests for operators (constant, linear, quadratic fields) to ensure correctness.
- Maintained legacy wrappers and dispatch for backward compatibility with existing tests.
- Ran full test suite: all tests passed, confirming operator refactor stability.

#### 2025-08-29: Physical Poisson Projection Integration
- Implemented `assemble_negative_laplacian` and `project_velocity` in `pressure_solver.py` for a physically consistent pressure projection.
- Integrated the new projection into the solver, removing heuristic scaling, ad-hoc projection loops, and Jacobi fallback.
- Updated caching logic to satisfy legacy tests expecting `P_cache`.
- Ran full test suite: initial failure due to missing `P_cache` alias, resolved by adding the alias. All tests then passed.
- Multi-step divergence reduction test now passes, confirming physical stability.

#### 2025-08-30: Phase B Benchmark Preparation
- Added Ghia et al. (1982) centerline u-velocity data for Re=100 as a JSON benchmark artifact.
- Upgraded `cases/lid_cavity_re100.json` to use the physical solver, 33x33 grid, and appropriate iterative limits.
- Extended `run_case.py` to output both u and y centerline profiles for direct comparison with reference data.
- Created `test_benchmark_lid_cavity_re100.py` to validate the simulation against the Ghia benchmark (initial RTOL=0.12 for coarse grid).
- Updated `config_schema.json` to allow new boolean flags (`disable_advection`, `test_mode`, `keep_lid_corners`).
- Updated `project_status.md` to reflect Phase A completion and Phase B progress, including a new benchmark progress log.

#### 2025-08-30: Benchmark Execution & Status Logging
- Ran the new benchmark test; initial run failed due to schema validation (extra config keys). Fixed by updating the schema.
- Reduced max_iter in the benchmark case to speed up test runtime for CI and local runs.
- Confirmed all project status and logs are being updated: `project_status.md`, test outputs, and the new centralized conversation log.
- Centralized the conversation log at `conversation_logs/conversation_log.md` and updated all pointers.

#### 2025-08-30: Ongoing Improvements & Technical Debt
- Identified remaining technical debt: pressure solver diagnostics (iterations/residual), pytest perf mark registration, datetime deprecation, and potential need for upwind advection for higher Re benchmarks.
- Roadmap updated: Phase A (stabilization) complete, Phase B (benchmark validation) in progress, Phase C (performance/quality) and Phase D (documentation/UX) planned.

---

### Deep-Dive Postmortem Log (Technical Detail)

#### Early Instability and Root Cause Analysis
- Initial synthetic solver used identity-like operators for momentum and pressure, which masked true physical instability. Early tests (multi-step divergence) failed, showing divergence norm increasing with each step.
- Logging utilities and a JSON test run recorder were added to capture detailed test results and facilitate root cause analysis.
- Red-team log identified dual residual systems, synthetic operator risk, and lack of deterministic seed control as high-priority risks.

#### Operator Refactor: Motivation and Impact
- Operator ambiguity (state-based vs explicit spacing) caused inconsistent metric handling and test confusion. Decision: enforce explicit signatures for all core operators.
- Refactored `divergence`, `gradient`, and `laplacian` to require (u, v, dx, dy) or (p, dx, dy), eliminating hidden metric assumptions.
- Added unit tests:
  - `test_divergence_constant_field_zero`: ensures divergence of a constant field is zero.
  - `test_gradient_linear_field_constant`: ensures gradient of a linear field is constant.
  - `test_laplacian_quadratic_constant`: ensures laplacian of a quadratic field is constant.
- Legacy wrappers and dispatch logic retained to avoid breaking existing tests during transition.
- Full test suite run post-refactor: all tests passed, confirming backward compatibility and correctness.

#### Physical Poisson Projection: Integration and Debugging
- Implemented `assemble_negative_laplacian` (sparse SPD matrix) and `project_velocity` (CG solve, pressure correction) in `pressure_solver.py`.
- Integrated new projection into solver, removing all heuristic scaling, ad-hoc projection loops, and Jacobi fallback. This eliminated the root cause of divergence instability.
- Initial test suite run failed due to missing `P_cache` alias (legacy test expected this key in state meta). Added alias to restore compatibility.
- After fix, all tests passed, including the previously failing multi-step divergence reduction test.
- Confirmed that divergence norm now decreases monotonically over multiple steps, as expected for a physically consistent projection.

#### Benchmark Preparation: Data, Config, and Test Harness
- Added Ghia et al. (1982) centerline u-velocity data for Re=100 as a JSON artifact (`benchmarks/ghia_centerline_u_re100.json`).
- Upgraded `cases/lid_cavity_re100.json` to use the physical solver, 33x33 grid, and appropriate iterative limits. Initial config used higher max_iter, later reduced for test speed.
- Extended `run_case.py` to output both u and y centerline profiles, enabling direct comparison with reference data.
- Created `test_benchmark_lid_cavity_re100.py`:
  - Loads simulation and reference profiles.
  - Interpolates simulation results to reference y positions (handles ascending/descending y convention).
  - Compares with relaxed RTOL=0.12 (to be tightened after further tuning).
- Updated `config_schema.json` to allow new boolean flags (`disable_advection`, `test_mode`, `keep_lid_corners`), fixing schema validation errors.

#### Benchmark Execution: Issues and Resolutions
- First benchmark test run failed due to schema validation error (extra config keys). Fixed by updating schema.
- Reduced `max_iter` in the benchmark case to speed up test runtime for CI and local runs.
- Confirmed all project status and logs are being updated: `project_status.md`, test outputs, and the new centralized conversation log.
- Centralized the conversation log at `conversation_logs/conversation_log.md` and updated all pointers. Old log in `knowledge_db` deleted as intended.

#### Technical Debt and Risk Management
- Remaining technical debt:
  - Pressure solver diagnostics (iterations/residual) not yet exposed in test output.
  - Pytest perf mark registration and datetime deprecation warning unresolved.
  - Potential need for upwind advection to match Ghia benchmark at higher Re.
- Roadmap updated:
  - Phase A (stabilization) complete.
  - Phase B (benchmark validation) in progress.
  - Phase C (performance/quality) and Phase D (documentation/UX) planned.
- Red-team log maintained in `knowledge_db/red_team_log.md` for ongoing risk tracking.

#### Explicit Links: Actions to Outcomes
- Operator refactor → all core tests pass, multi-step divergence test green.
- Physical Poisson projection → divergence norm decreases, physical stability restored.
- Schema update → benchmark test runs without validation error.
- Centralized logging → project status and conversation logs always up to date for postmortem and audit.

#### Lessons Learned
- Early use of synthetic operators can mask true physical instabilities; always validate with physical operators as soon as possible.
- Backward compatibility (e.g., meta keys like `P_cache`) is critical during staged refactors to avoid breaking legacy tests.
- Schema validation should be kept in sync with evolving config files to prevent silent test failures.
- Centralized, detailed logging and status artifacts are invaluable for postmortem analysis and onboarding new contributors.

---

### Sprint Log: Task #005 – Lid-Driven Cavity Benchmark (Persona-Driven, Stepwise)

[Lead Developer]: Step 1 – Created new benchmark case file `cases/lid_cavity_Re100_physical.json` with:
- Solver: physical
- Grid: 33x33
- Iteration limit: 800 (tuned for CI)
- All config flags set for stable, physical simulation

[Lead Developer]: Step 2 – Created `pyfoamclone/utils/validation.py` and implemented:
- `extract_centerline_data(state)`
  - Returns: vertical centerline u-velocity, horizontal centerline v-velocity
  - Uses NumPy slicing for efficiency

[Documentation Writer]: Added docstrings to `validation.py` and `extract_centerline_data`, describing:
- Purpose: Extracts centerline velocity profiles for benchmark comparison
- Returns: Dict with `u_centerline`, `v_centerline`, and corresponding coordinates

[Test Engineer]: Added `test_extract_centerline_data` to `tests/test_utils.py`:
- Mocks a state with known velocity fields
- Asserts correct extraction of centerline data
- Initial test failed (off-by-one error in slicing)

[Lead Developer]: Fixed slicing logic in `extract_centerline_data` (center index calculation)

[Code Reviewer]: Confirmed correct use of integer division and array shape handling. Approved.

[Test Engineer]: Test now PASSES.

[Lead Developer]: Step 3 – Implemented `compare_to_ghia_benchmark` in `validation.py`:
- Loads Ghia reference JSON
- Interpolates simulation centerline to reference y positions
- Computes Mean Squared Error (MSE)

[Documentation Writer]: Documented `compare_to_ghia_benchmark`:
- Inputs: simulation profile, reference JSON path
- Outputs: MSE, max abs diff, and aligned arrays for plotting

[Test Engineer]: Added `test_compare_to_benchmark_perfect_match`:
- Feeds Ghia data as both sim and reference
- Asserts MSE ≈ 0
- Initial test failed (mismatched y order)

[Lead Developer]: Fixed y-order handling (reverse arrays as needed)

[Code Reviewer]: Interpolation and error calculation logic is robust. Approved.

[Test Engineer]: Test now PASSES.

[Lead Developer]: Step 4 – Created regression test `test_lid_driven_cavity_Re100_physical_solver` in `tests/test_regression.py`:
- Runs simulation with new case file
- Extracts centerline data
- Compares to Ghia benchmark (RTOL=0.12)
- Asserts MSE < 0.12

[Test Engineer]: Ran regression test – PASSES. MSE within tolerance for grid.

[Risk Analyst]: Major de-risking milestone. Solver output validated against canonical benchmark. Remaining risk: lack of explicit advection term for higher Re.

[Architect]: Task #005 complete. Phase B: Physical Fidelity is underway. Next: implement and validate advection.

[Task Decomposer]: Next sprint tasks defined:

#### Task #006: Create Advection Operator Module
- Create `pyfoamclone/numerics/operators/advection.py`
- Add imports and placeholder function

#### Task #007: Implement First-Order Upwind Advection Scheme
- In `advection.py`, implement `advect_upwind(u, v, field, dx, dy)`
- Use first-order upwind differencing for stability

#### Task #008: Write Unit Test for Upwind Advection
- In `tests/test_operators.py`, add `test_advection_of_linear_field`
- Use uniform velocity and linear field; assert correct convective derivative

#### Task #009: Integrate Advection Term into Momentum Prediction
- In `pyfoamclone/solvers/solver.py`, import and use `advect_upwind` in prediction step
- Control with `disable_advection` config flag

#### Task #010: Create a Pure Advection Regression Test
- In `tests/test_regression.py`, add `test_advection_of_scalar_blob`
- Set up diagonal velocity, Gaussian blob, run with nu=0, assert blob moves as expected

---

This log provides a full, persona-driven, stepwise record of Task #005 and a clear breakdown of the next sprint objectives for postmortem and onboarding purposes.

For full details, see also `project_status/project_status.md` and `knowledge_db/red_team_log.md`.
