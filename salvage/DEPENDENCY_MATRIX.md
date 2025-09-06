# Dependency & Interface Matrix (v0.1)

Layer | Provides | Depends On | Stable Interfaces (Contracts) | Anti-Patterns to Avoid
----- | -------- | ---------- | ----------------------------- | ----------------------
config | Parsed, validated config object | (None) | get_config(), schema version | Business logic in config parsing
core.structures | Grid, Field, BCDescriptor | config | Grid(nx, ny, spacing), Field(name, data, centering) | Coupling logic in structures
math.operators | Discrete gradients, divergence, laplacian | core.structures | op_gradient(field, grid), op_div(field, grid) | Direct file IO, global state
assembly | System matrix and RHS build | math.operators, core.structures | assemble_momentum(...), assemble_pressure(...) | Embedding solver steps inline
coupling | SIMPLE/PISO orchestration | assembly, timestep, linalg, utils | simple_step(state), piso_step(state) | Holding long-lived mutable solver state across runs
linalg | Linear solves & preconditioners | assembly | solve_pressure(A, b, tol, max_iter) | Coupling back-calls (circular)
timestep | Adaptive dt control | utils.residuals, core.structures | next_dt(residuals, prev_dt, CFL_target) | Accessing raw matrix internals
physics | Turbulence, scalar transport | core.structures, math.operators | model.apply(state) | Mutating unrelated fields silently
io | Serialization, checkpoint, VTK output | core.structures | write_vtk(fields, grid, path) | Encoding solver logic
utils | Residual calc, logging, profiling | (Minimal) | compute_residual(state), log_failure(...) | Hidden numerical ops
reuse | Salvaged utility functions | utils | Reused validated abstractions | Divergent copies of same logic
cli | Entry points | All higher abstractions | main(argv) launching configured case | Deep algorithm code
telemetry | Structured event emission | utils, coupling | emit_residual(record) | Format drift without version bump
governance | Gates: complexity, duplication | utils, telemetry | run_precommit_checks(changes) | Silent auto-fixes

Stability Policy: Lower layers must not import higher. Violation triggers dependency graph gate failure.
