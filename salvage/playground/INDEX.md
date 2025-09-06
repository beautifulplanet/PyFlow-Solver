# Playground ↔ Chapters ↔ Workflow Gates

Map of playground scripts to textbook chapters and workflow gates for quick study and validation.

- manufactured_solution_order.py
  - Chapters: 22 (Discrete Operators), 26 (Manufactured Solutions)
  - Gates: P2 Operator verification
- adaptive_dt_pid_sim.py
  - Chapters: 25 (Time Integration), 27 (Residual Dynamics)
  - Gates: P4 Stabilization
- residual_plateau_classifier_demo.py
  - Chapters: 27 (Residual Dynamics), 20 (KPI Dashboards)
  - Gates: P4 Plateau diagnostics
- stencil_symmetry_audit.py
  - Chapters: 22 (Operators), 23 (Assembly)
  - Gates: P2 Symmetry/consistency audit
- config_schema_validator.py
  - Chapters: 6 (Dataset Design), 16 (Lifecycle), 56–60 (Release & Compliance)
  - Gates: P0 Config hygiene
- duplication_similarity_demo.py
  - Chapters: 14 (Duplicate Pipeline), 51–53 (Semantic/Embedding)
  - Gates: Governance duplication guard
- microbench_kernels.py
  - Chapters: 31–34 (Performance)
  - Gates: P5 Baselines
- embedding_similarity_demo.py
  - Chapters: 52 (Embedding), 53 (Developer UX)
  - Gates: Governance duplication guard (embedding mode)
- failure_injection_harness.py
  - Chapters: 37–40 (Reliability & Resilience)
  - Gates: Nightly coverage
- multi_objective_tuner_demo.py
  - Chapters: 28 (Auto-Tuners), 54 (Multi-Objective)
  - Gates: Tuning feasibility
- aggregate_dashboard.py, render_dashboard_md.py
  - Chapters: 44–45 (Report Cards & Comparative Analytics)
  - Gates: Reporting pipeline
- run_fast_checks.py
  - CI fast path for S1–S3 gates
