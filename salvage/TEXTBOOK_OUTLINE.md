# CFD Intelligence & Solver Engineering: From Legacy Corpus to Production System

Subtitle: A Comprehensive Guide to Postmortem Mining, Knowledge Distillation, Governance, and High‑Reliability CFD Solver Construction

Authoring Blueprint Version: 0.1 (Pre‑Manuscript)

## Pedagogical Vision
- Audience Tiers: (A) Senior CFD developers, (B) Scientific software engineers, (C) ML/AI for engineering researchers.
- Learning Modalities: Concept exposition, procedural playbooks, metric dashboards, forensic case studies, guided experiments, failure drills.
- Recurrent Pillars: (1) Observability, (2) Reusability, (3) Numerical Integrity, (4) Governance, (5) Continuous Validation, (6) Evolution & Drift Management.

## Macro Structure
| Part | Theme | Core Question | Outcome Artifact |
|------|-------|--------------|------------------|
| I | Corpus Forensics | What is in the legacy code & why did it fail/succeed? | Postmortem Report & Taxonomy |
| II | Data & Knowledge Engineering | How do we convert code history into structured, queryable intelligence? | Learning Dataset & Knowledge DB |
| III | Salvage & Reuse Architecture | How do we extract, score, and redeploy high-quality fragments safely? | Reusable Library & Reuse Policy |
| IV | Governance & Guardrails | How do we prevent regression & drift? | Guardrail Suite & KPIs |
| V | Numerical Core Construction | How to architect a modern modular CFD solver? | Baseline Solver Design Spec |
| VI | Stability & Accuracy Engineering | How to guarantee convergence & fidelity? | Convergence Playbooks & Validation Matrix |
| VII | Performance & Scalability | How to attain & sustain performance? | Optimization Ledger & Roofline Maps |
| VIII | Reliability & Resilience | How to survive failure modes gracefully? | Failure Catalog & Resilience Score |
| IX | Telemetry & Analytics | How to observe and reason about runs at scale? | Unified Telemetry Schema & Dashboards |
| X | Advanced Physics & Extensions | How to extend into turbulence, scalars, compressibility? | Plug‑in Contracts & Extension Templates |
| XI | ML & Semantic Intelligence | How to integrate AI retrieval and embedding systems responsibly? | Semantic Index + Duplicate Guard |
| XII | Release, Compliance, Security | How to ship and attest a trustworthy solver? | Release Attestation Bundle |
| XIII | Future Frontiers | What next (UQ, adaptive meshes, hybrid ML)? | Research Roadmap |

## Detailed Chapter Outline

### Part I: Corpus Forensics
1. The Rationale for Postmortem Mining
   - Legacy risk, duplication debt, silent divergence.
   - Case vignette: Repeated divergence fix anti-patterns.
2. Building a Workspace Inventory
   - Directory scanning heuristics; filtering third‑party noise.
   - Metrics: file density, extension entropy, age stratification.
3. Failure & Success Taxonomy Engineering
   - Label inference heuristics (filename tokens, keywords, AST anomalies).
   - Creating consistent failure reason clusters; similarity metrics.
4. Evolution Chain Reconstruction
   - Prototype→Final linking via token stem & semantic diff.
   - Evolution states (proto, mature, legacy, orphan).
5. Forensic Reporting
   - Synthesizing narrative vs metric table synergy.
   - Artifact spec: `postmortem_report.md` schema.

### Part II: Data & Knowledge Engineering
6. Learning Dataset Design
   - JSONL entry schema: structural, semantic, failure, remediation.
   - Quality labeling heuristics & noise handling.
7. Function-Level Deep Extraction
   - AST analysis: complexity proxies, import role classification, salvage scoring.
8. Dependency Graph & Centrality
   - Graph construction; centrality vs refactor priority correlation.
9. Knowledge Base & Semantic Index
   - Vocab building, TF‑IDF weighting, forward upgrade path to embeddings.
10. SQLite Consolidation & Query Templates
   - Schema for functions/files/metadata; example analytic queries.

### Part III: Salvage & Reuse Architecture
11. Scoring & Selecting High-Value Code
   - Salvage score formula; balancing complexity and uniqueness.
12. Normalizing & Hardening Salvaged Snippets
   - Strip side effects, parameterize constants, add invariants.
13. Reuse Module Curation Workflow
   - Promotion checklist; provenance annotation.
14. Duplicate Detection Pipeline
   - Token normalization; similarity thresholds; override governance.
15. Reuse Effectiveness Metrics
   - Measuring reduction in new LOC & defect rates.

### Part IV: Governance & Guardrails
16. Evolution Tagging & Lifecycle States
   - Tag semantics; transitions & enforcement.
17. Complexity & Size Gates
   - Pre-commit analyzer design, thresholds & escalation.
18. Failure Logging & Structured Telemetry Hooks
   - JSON schema evolution; hash chain integrity.
19. Promotion Gate Architecture
   - Multi-metric gating (residual, accuracy, complexity, duplication, performance).
20. Drift Monitoring & KPI Dashboards
   - Complexity drift, reuse ratio, failure rate graphs.

### Part V: Numerical Core Construction
21. Grid & Field Design Patterns
   - Ghost layer strategies; memory footprint budgeting.
22. Discrete Operators & Stencils
   - Accuracy/consistency checks, symmetry tests, dispersion analysis.
23. Linear System Assembly & Storage
   - Sparse formats trade-offs; structural hashing.
24. Pressure–Velocity Coupling Strategies
   - SIMPLE vs PISO; Rhie–Chow; oscillation detection.
25. Time Integration & Adaptive Step Control
   - PID dt controller, rollback design, plateau classification.

### Part VI: Stability & Accuracy Engineering
26. Manufactured Solutions & Order Verification
27. Residual Dynamics & Plateau Diagnostics
28. Under-Relaxation Tuning & Auto-Tuners
29. Conservation Law Monitoring
30. Benchmarks Corpus Construction (multi-tier)

### Part VII: Performance & Scalability
31. Microkernel Benchmarking & Normalization
32. Roofline & Bottleneck Attribution
33. Vectorization & Memory Bandwidth Tactics
34. Solver Escalation Ladder & Preconditioner Strategy
35. Parallelization Roadmap (MPI/GPU staging plan)

### Part VIII: Reliability & Resilience
36. Failure Mode Catalog & Taxonomy Coverage
37. Failure Injection Harness Design
38. Snapshot Minimization & Atomic Checkpointing
39. Recovery & Degradation Modes
40. Resilience Score Composition (R-score)

### Part IX: Telemetry & Analytics
41. Unified Telemetry Schema & Versioning
42. Online Aggregation & Rolling Statistics
43. Anomaly Detection (EWMA, Z-score, ML prototype)
44. Benchmark Report Card Automation
45. Cross-Run Comparative Analytics & Trend Surfaces

### Part X: Advanced Physics & Extensions
46. Turbulence Model Plug-in Interface
47. Scalar Transport & Source Term Abstractions
48. Compressible Flow Branching Strategy
49. Transitional & Multiphase Considerations (roadmap)
50. Extension Test Harness & Contract Tests

### Part XI: ML & Semantic Intelligence
51. Semantic Index Foundations & Calibration
52. Embedding Upgrade & Similarity Drift Management
53. Duplicate Detection Governance & Developer UX
54. Auto-Tuning via Multi-Objective Optimization
55. Hybrid ML Surrogates (e.g., learned closure fields) – guardrails

### Part XII: Release, Compliance, Security
56. SBOM, Supply Chain Integrity & Attestation
57. Security Scans & Secret Hygiene in Scientific Code
58. Release Promotion Checklists & Rollback Strategy
59. Licensing & IP Considerations in Salvage
60. SLA & KPI Reporting Automation

### Part XIII: Future Frontiers
61. Uncertainty Quantification (sampling, polynomial chaos) integration
62. Adaptive Mesh Refinement Governance
63. Bayesian Calibration of Model Parameters
64. Energy Efficiency & Green Computing Metrics
65. Research Roadmap & Open Challenges

## Chapter Template (Micro-Level Structure)
1. Motivation / Pain Point
2. Conceptual Framework
3. Formal Definitions / Schemas
4. Algorithms & Pseudocode (no production code yet)
5. Metrics & Validation Criteria
6. Instrumentation & Observability Hooks
7. Failure Points & Mitigations
8. Governance & Policy Interactions
9. Worked Case Study (mini dataset / run scenario)
10. Exercises (Hands-on tasks & reflection)
11. Further Reading

## Cross-Cutting Artifacts & Diagrams
- Layered Architecture Diagram (Parts V & XI interplay).
- Data Flow: Corpus → Learning Dataset → Knowledge DB → Semantic Index → Governance Hooks.
- Convergence Lifecycle Timeline (residual families, plateau detection, tuner adaptation).
- Performance Pipeline (microkernel benchmarking to roofline to regression classification).
- Failure Handling Flow (detection → classification → snapshot → root cause → remediation).
- Promotion Gate Decision Graph.

## Metrics Catalogue (Canonical Definitions)
- Provide unified glossary: residual orders, drift %, reuse ratio, centrality score, salvage score, resilience score, performance normalization factor, duplication similarity, stability envelope.

## Benchmark Dataset Specification
| Benchmark | Domain | Key Metric | Reference Source | Tolerance Tier |
|-----------|--------|-----------|------------------|----------------|
| Cavity Re=100 | 64² | Centerline u | Ghia et al. | A |
| Cavity Re=5000 | 256² | Centerline u | Lit. comp. | B |
| Channel Laminar | 128×64 | Cf | Blasius | A |
| TGV 2D | 128² | Energy decay RMS | Analytical | A |
| Scalar Rotating Vortex | 128² | L2 error | Manufactured | A |
| Backward Step | Var | Reattach length (x_r) | Literature | B |
| Rayleigh–Bénard | 128² | Nu | Reference DNS | B |
| Cylinder Low Re | 256² | St | Experimental | C |
| Shock Tube | 400 | Shock pos | Analytical | A |

## Multi-Objective Parameter Tuning Framework
- Decision Vector: relaxation_u, relaxation_p, dt_safety, preconditioner_type.
- Objective Set: [ ConvergenceTime, FinalResidual, StabilityFailures, AccuracyError ].
- Pareto Frontier Archiving & Promotion Criteria.

## Governance KPI Dashboard Fields
| KPI | Source | Update Cadence | Alert Threshold |
|-----|--------|----------------|-----------------|
| Residual Orders | Telemetry | per run | < target -0.5 |
| Complexity Drift | Manifest diff | weekly | > +10% |
| Reuse Ratio | Diff analyzer | weekly | < 0.30 |
| Performance Regress | Microbench | per PR | >5% time increase |
| Accuracy Regress | Benchmark suite | nightly | > tolerance |
| Unclassified Fail | Failure logs | per run | >0 |
| Index Freshness | Commit count | daily | >10 commits stale |

## Failure Injection Matrix (Expanded)
| ID | Injection | Layer | Expected Detection | Classification |
|----|----------|-------|--------------------|----------------|
| FI1 | NaN in velocity interior | operators | Per-step NaN scan | F3 |
| FI2 | Wrong sign in Laplacian coeff | assembly | Symmetry norm fail | F?->F1 |
| FI3 | Dropped boundary condition | bc | Mass imbalance | F4 |
| FI4 | Stale dt reuse (no adapt) | timestep | Plateau + dt variance | F2 |
| FI5 | Corrupted pressure RHS | coupling | Oscillation sentinel | F1 |
| FI6 | Silent data truncation (float32 cast) | numerics | Accuracy diff test | F6 |
| FI7 | Snapshot partial write | reliability | Checksum mismatch | F? |
| FI8 | Duplicate logic insertion | governance | Similarity gate | Governance event |

## Pedagogical Exercises Examples
- Exercise 1 (Part I): Given anonymized file stats, cluster failure reasons and propose three remediation heuristics; evaluate precision/recall using hidden answer key.
- Exercise 2 (Part III): Compute salvage scores for five function ASTs; justify inclusion or exclusion.
- Exercise 3 (Part V): Derive central differencing stencil truncation error and design a unit test harness to verify order numerically.
- Exercise 4 (Part VI): Implement theoretical plateau classifier pseudocode and analyze false positive scenarios.
- Exercise 5 (Part VII): Build a roofline estimate from microkernel timings and identify memory vs compute bound regions.
- Exercise 6 (Part XI): Calibrate semantic similarity thresholds using provided function corpus pairs; produce ROC curve and select operating point.

## Research Extensions Roadmap
- Hybrid Data‑Driven Closure Models with Confidence Gating.
- Adaptive Mesh Refinement with Budgeted Complexity Drift Monitoring.
- Domain Decomposition Load Balancing via Telemetry Feedback Loop.
- Uncertainty Quantification Pipeline (Latin Hypercube + Surrogate reuse).
- Energy & Carbon Footprint Instrumentation (performance telemetry extension).

## Writing & Style Guidelines
- Terminology Consistency: maintain controlled vocabulary (append glossary).
- Code Snippets (Later): must trace to validated production pattern; highlight diff vs naive anti-pattern.
- Figures: all diagrams vector (SVG) with colorblind-safe palette.
- Equations: accompany each with variable table & units.
- Case Studies: real metrics (sanitized) + root cause narrative + quant lessons.

## Open Content Gaps (To Flesh Out in Manuscript Phase)
- Formal salvage scoring mathematical derivation & statistical validation.
- Empirical duplicate detection precision/recall study.
- Longitudinal complexity drift dataset (multi-month simulation).
- Comparative study of CFL controllers (PID vs heuristic shrink/expand).
- Multi-objective tuner algorithmic complexity & convergence properties.

## Manuscript Production Plan (High-Level)
| Sprint | Chapters | Draft Type | Exit Criteria |
|--------|----------|------------|---------------|
| 1 | 1–5 | Exploratory | Structural outline & diagrams complete |
| 2 | 6–10 | Alpha | 80% prose, pending figures |
| 3 | 11–15 | Alpha | Salvage & governance examples validated |
| 4 | 16–20 | Beta | Metrics glossary stable |
| 5 | 21–25 | Beta | Numerical proofs internally reviewed |
| 6 | 26–30 | Beta | Benchmark table finalized |
| 7 | 31–35 | Beta | Performance case study inserted |
| 8 | 36–40 | RC1 | Failure injection appendix done |
| 9 | 41–45 | RC1 | Telemetry schema version locked |
| 10 | 46–50 | RC1 | Extension interfaces validated |
| 11 | 51–55 | RC2 | Semantic threshold calibration data |
| 12 | 56–60 | RC2 | Security & release bundle example |
| 13 | 61–65 | RC2 | Future roadmap consensus |
| 14 | All | Final | Copy edit & technical review passed |

## Review & Quality Gates for Book Content
- Technical Accuracy Reviewers: separate domain experts per Part.
- Consistency Linter: automated check for glossary term usage.
- Metrics Reproducibility: scripts produce each figure/table from raw artifact snapshots.
- Anti-Entropy Check: ensure no chapter drifts from canonical definitions file.

## Appendices (Planned)
A. Glossary & Controlled Vocabulary
B. Metrics Formal Definitions (mathematical)
C. Telemetry Schema Versions & Migration Log
D. Failure Taxonomy Reference
E. Benchmark Reference Data Sources & Citations
F. Sample Decision Log Entries
G. Salvage Score Calibration Study Protocol
H. Semantic Similarity Calibration Dataset Summary
I. Roofline Modeling Walkthrough
J. Promotion Gate Script Pseudocode

---
End of Outline v0.1. Iterate before manuscript drafting. Update CHANGELOG on each structural modification.
