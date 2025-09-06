# Glossary & Controlled Vocabulary (v0.1)

Term | Definition | Notes
---- | ---------- | -----
Adaptive dt Controller | Algorithm adjusting timestep to maintain target CFL or error bounds | Uses PID-like scaling
Baseline Case | Canonical simple configuration used for regression and comparison | E.g., 2D Lid Cavity Re=100
Benchmark Corpus | Curated set of validation scenarios with references & tolerances | Tiered A/B/C targets
Centrality Score | Graph metric quantifying structural influence of a module/function | Used to prioritize refactors
CFD Promotion Gate | Composite criteria required before merging solver changes to main | Convergence, accuracy, perf, complexity
Complexity Drift | Relative increase in structural complexity vs baseline snapshot | Trigger for refactor planning
Conservation Error | Deviation from zero for conserved quantity (mass, energy) | Expressed as relative measure
Coupling Oscillation Sentinel | Detector for checkerboard or periodic pressure/velocity artifacts | Threshold amplitude based
Decision Log | Append-only ledger of architectural decisions & rationale | Includes reversal cost rating
Duplicate Similarity | Cosine similarity between new code and existing corpus vector | TF-IDF or embedding space
Evolution Tag | Lifecycle label: proto, mature, legacy, orphan | Drives salvage priority
Failure Injection | Synthetic perturbation to validate detection & resilience | Cataloged by ID (FI1...)
Failure Taxonomy | Structured classification of failure modes with codes | Drives coverage metrics
Golden Metric | Reference value for validation metric plus tolerance | Stored with hash & provenance
Halo / Ghost Layer | Additional cell layer around domain for stencil operations | Needs consistent synchronization
Hash Manifest | File hash inventory establishing reproducible state | Supports drift detection
Manufactured Solution | Analytical field constructed to validate discretization order | Injected source term
Microkernel Benchmark | Isolated performance test of a numerical primitive | Normalized by cell count
Mixed Precision | Use of different floating-point precisions within solver | Requires accuracy guardrails
Pareto Frontier | Set of non-dominated parameter configurations in multi-objective tuning | Guides auto-tuner choices
Plateau Classification | Identification of convergence stagnation via residual slope | Supports adaptive strategy
Residual Orders Drop | log10(r0 / rN) convergence indicator | Gate threshold ≥ 3 typically
Resilience Score (R-score) | Composite reliability metric incorporating failure handling | Weighted incident classes
Reuse Ratio | Salvaged LOC reused / new LOC added | Measures effectiveness of salvage program
Rhie–Chow Interpolation | Technique preventing pressure-velocity decoupling | Applied in SIMPLE/PISO
Salvage Score | Heuristic quality metric for function reuse candidacy | Complexity, uniqueness, reliability
Semantic Index | Vector index enabling similarity & reuse recommendations | TF-IDF base, embedding upgrade path
Stencil Symmetry Norm | Norm of (A - A^T) for discretized operator matrix | Should approach 0 for symmetric operators
Telemetry Schema Version | Incrementing version for structured run data | Enables migrations
Under-Relaxation Factor | Multiplicative dampening applied to iterative updates | Affects stability & speed
Validation Tolerance Tier | A (stringent), B (moderate), C (lenient) accuracy classification | Drives gating

(End v0.1; extend as new controlled terms arise.)
