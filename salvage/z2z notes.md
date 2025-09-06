# Z2Z Notes: AI-Governed CFD Program – Data Correlation and 7‑AI Committee Operating Model

Purpose
- Convert the postmortem corpus, learning datasets, knowledge base, and playground outputs into a governance plan for a 7‑AI committee that filters ideas for accuracy, prevents drift/errors, and enforces scope so we avoid over/under build.
- This report assumes all scaffolding was AI‑generated; controls below address quality and safety.

Source Artifacts Considered
- Postmortem + scans: workspace inventories, success/failure taxonomy, evolution chains.
- Datasets: learning_dataset.jsonl (per‑file), extended_learning_dataset.jsonl (function‑level), deeper_metrics.json.
- Salvage/Reuse: orphan_salvage.jsonl, reusable_funcs.py, refactor_plan.md.
- Knowledge: knowledge_db/*.jsonl, cfd_knowledge.db (SQLite), tfidf_index.json; semantic_query.py.
- Guardrails: PRECOMMIT_COMPLEXITY_REPORT.md, generate_precommit_hook.py, failure_logging_helper.py.
- Planning: WORKFLOW_CFD_SOLVER_PLAN.md, WORKFLOW_CFD_REFINEMENTS_V23_V122.md, TEXTBOOK_OUTLINE.md.
- Playground: manufactured order, plateau classifier, adaptive‑dt PID, stencil symmetry audit, duplication & embedding demos, failure injection, microbenchmarks, multi‑objective tuner; dashboard_summary.{json,md}.

Key Correlations to Future Project
- High salvage density in failing files → committee should favor reuse over fresh code; require similarity scan before new operator code.
- Warnings (deprecated/timezone, invalid escapes) previously polluted metrics → enforce schema and linter gates pre‑merge.
- Centrality analysis isolated refactor hotspots → architecture committee must split monoliths early; size gates prevent relapse.
- Residual plateaus frequent → standardize plateau classifier and adaptive‑dt PID as first‑class gates.
- Third‑party noise skewed earlier metrics → strict directory exclusions and manifest hashing required.
- TF‑IDF index proved useful for deduplication → upgrade path to embeddings with calibration set to prevent threshold drift.

7‑AI Committee: Roles, Inputs, and Gates
1) Accuracy & Physics (A1)
   - Verifies benchmark/analytic targets; runs manufactured solution and case profiles.
   - Inputs: benchmark references, playground manufactured results, golden metrics.
   - Gate: residual drop ≥3 orders on cavity; error ≤ tier thresholds.
2) Numerical Stability (A2)
   - Reviews schemes, plateaus, dt PID settings; approves rollback/step‑reject logic.
   - Inputs: plateau classifier, failure injection results, adaptive‑dt simulations.
   - Gate: no divergence in stress matrix; plateau misclassifications <5% on synth set.
3) Performance & Memory (A3)
   - Ensures kernel normals and baselines; flags regressions >5%.
   - Inputs: microbench JSON, flamegraph notes (later), time_per_cell_us trends.
   - Gate: ≥2× vs naive baseline per P5; memory growth <10%.
4) Architecture & Complexity (A4)
   - Enforces layered deps, function LOC caps, and refactor plan adherence.
   - Inputs: dependency graph deltas, complexity snapshots, refactor_plan.md.
   - Gate: max func LOC ≤60 (no waiver); no upward imports; monolith split schedule.
5) Reuse & Duplication (A5)
   - Mandates salvage‑first; blocks near‑duplicates.
   - Inputs: TF‑IDF/embedding sims, knowledge_db lookups, reusable_funcs references.
   - Gate: duplicate sim >0.90 blocked unless DECISION file justifies.
6) Security & Compliance (A6)
   - Scans SBOM, licenses, secret hygiene, provenance of salvage.
   - Inputs: SBOM, license map, secret scans.
   - Gate: 0 critical findings; provenance annotations present.
7) Product & Scope (A7)
   - Applies scope guardrails; checks YAGNI; prioritizes MVP increments.
   - Inputs: roadmap, backlog, KPI dashboard; impact estimates.
   - Gate: aligns to current phase; over/under build score within tolerance.

Idea Intake → Single‑Idea Filter Process
- Intake: Each feature/idea submitted with: problem statement, expected metric deltas, affected layers, reuse candidates, risk score.
- Scoring (per committee):
  - A1 Accuracy (0–5), A2 Stability (0–5), A3 Perf (0–5), A4 Complexity (0–5), A5 Reuse (0–5), A6 Security (0–5), A7 Scope (0–5).
  - Weighted composite: W = 0.25*A1 + 0.20*A2 + 0.15*A3 + 0.15*A4 + 0.10*A5 + 0.10*A7 + 0.05*A6.
- Decision Protocol:
  - Minimums: A1≥4, A2≥4, A7≥3; any hard gate fail → reject or revise.
  - Ties resolved by risk‑adjusted value = W / (1 + R‑score), where R‑score from failure history (F1–F20) likelihood.
  - Outcome: exactly one idea selected per cycle; others queued with remediation notes.

Drift & Error Prevention (Operational)
- Telemetry & Golden Metrics
  - Residual orders, centerline errors, energy decay errors stored per run; golden diffs checked per PR.
- Schema Versioning & Migration
  - Telemetry schema_version; migrations gated by A4/A6 sign‑off; historical compatibility tests.
- Drift Monitor
  - Weekly complexity drift ≤10%; duplication incidents = 0; index freshness <10 commits stale.
- Failure Injection Coverage
  - F1–F20 matrix nightly; 0 unclassified; report gaps to A2/A4.
- Reuse Enforcement
  - Pre‑commit similarity check (TF‑IDF/embedding); auto‑suggest salvage.
- Scope Control
  - Phase gates (P0–P9); idea rejected if it jumps phases or adds non‑critical deps.

Avoid Over/Under Build
- Overbuild Controls
  - Feature flags; interface stubs with contract tests before full implementation.
  - MVP acceptance criteria tied to KPIs; defer advanced physics until P5 passes.
- Underbuild Controls
  - Minimum strength: manufactured solution + cavity Re=100/1000 passing before expansion.
  - Benchmarks corpus tiering ensures adequate coverage before promotion.

Example Committee Walkthrough (QUICK Interpolation Idea)
- Inputs: Accuracy ↑ (staircase reduction), Stability ↔/↓ (risk of oscillations), Perf ↓ (more ops), Complexity ↑ (new path), Reuse ✓ (existing stencil variants), Scope ✓ (P2/P3 relevant).
- Scoring outcome: A1=4, A2=3, A3=3, A4=3, A5=4, A6=5, A7=4 → fails A2 minimum → action: run oscillation sentinel on sandbox; if pass, resubmit.

Runbooks & Cadence
- PR Path (S1–S4): schema lint, fast unit ops (playground fast checks), tiny cavity run, microbench; <6 minutes target.
- Nightly: full playground matrix + failure injections; dashboard refresh.
- Weekly: drift report; committee meeting auto‑generated agenda from incidents and KPIs.

Committee Prompts (Outline)
- A1: “Evaluate accuracy impacts and benchmark deltas; produce accept/reject with golden diff.”
- A2: “Analyze stability risk; run plateau classifier and failure injection subset; return F‑class risks.”
- A3: “Compare microbench normalized cost; flag regressions.”
- A4: “Check deps and complexity; list violations and refactor requirements.”
- A5: “Search knowledge_db for reuse; report top matches and duplication risks.”
- A6: “Scan SBOM/secrets; confirm provenance annotations.”
- A7: “Assess scope/phase alignment and user impact; rate over/under build risk.”

KPIs & Thresholds (Initial)
- Convergence: residual orders ≥3.0 on cavity 64².
- Accuracy: tier A/B/C per benchmark table (see workflow plan).
- Performance: ≥2× naive, aim 3.5× medium‑term.
- Complexity: max func LOC ≤60; drift ≤10%/release.
- Duplication: 0 blocking; warn at ≥0.85 similarity.
- Reliability: 0 unclassified failures; resilience score trending up.

Interfaces to Existing Assets
- Playground → Dashboard → Report Cards: used as A1/A2/A3 evidence.
- Knowledge DB + TF‑IDF/Embeddings: A5 duplication/reuse source.
- Pre‑commit Hook: A4/A5 enforcement.
- Failure Logging Helper: feeds F1–F20 coverage analytics.

Risk & Escalation
- Release Risk R = 0.4*AccuracyDelta + 0.2*PerfDelta + 0.15*RobustIncidents + 0.15*Drift + 0.1*SecurityAlerts → block if >0.65.
- Any Phase gate hard fail escalates to “revise or rollback” with incident stub and owner.

Deliverables per Idea Approval
- Decision log entry; design brief; test plan (benchmarks + failure injections); reuse mapping; KPI expectations; rollback plan.

Appendix: Mapping Failures to Committee Ownership
- F1/F2/F3 (divergence/plateau/NaN): A2 primary, A1 secondary.
- F5 (perf regression): A3.
- F6 (accuracy regression): A1.
- F11 (duplication): A5.
- F13 (telemetry schema drift): A4/A6.
- Security/Compliance (F19/F20): A6.
- Scope overruns: A7.

Summary
- The corpus and tooling support a disciplined, AI‑reviewed path to select exactly one accurate, stable, scoped idea per cycle. The committee roles, gates, and telemetry‑backed decisions prevent drift, reduce duplication, enforce architecture, and keep focus on validated physics and performance.
