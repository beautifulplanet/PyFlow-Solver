# Failure Catalog (v0.1)

ID | Name | Description | Detection Signal | Telemetry Fields | Mitigation | Residual Signature
-- | ---- | ----------- | ---------------- | ---------------- | ---------- | -----------------
F1 | Numerical Divergence | Residuals explode or become NaN/Inf | res_u > 10*r0 or NaN | step,res_*,dt,CFL_max | Reduce dt, increase relaxation, switch solver ladder | Sharp upturn then break
F2 | Plateau Stagnation | Residual slope ~0 for window | slope > -0.1 over W steps | slope,window_len | Adjust relaxation, refine mesh, switch scheme | Flat tail
F3 | NaN Contamination | Introduced NaNs without early termination | any NaN in field | field_hash, nan_count | Bisection isolation, add finite checks earlier | Sudden invalid
F4 | Mass Imbalance | Accumulated continuity residual > tol | continuity > tol_c | continuity, mass_flux | Tighten solver tolerance, refine pressure correction | Drift off zero
F5 | Performance Regression | Kernel time > baseline +5% | time_norm delta | kernel, time_norm | Investigate hotspot, optimize memory access | Elevated plateau
F6 | Accuracy Regression | Validation metric error > tolerance | error_rel > tol | metric,error_rel,ref | Revert change, tune parameters | Case-specific
F7 | Pressure Oscillation | Alternating pattern persists | pressure_osc_amp > thresh | osc_amp, freq | Enable Rhie–Chow, damping | Even-odd pattern
F8 | Solver Stagnation (Linear) | Linear solver residual constant | linear_res_slope ≈ 0 | lin_it, lin_res | Switch preconditioner, escalate solver | Flat micro residual
F9 | Drift in dt | dt variance high, oscillatory | var(dt)/mean > 0.2 | dt,var_dt | Smooth PID gains | High variance
F10 | Memory Leak | RSS grows linearly over long run | rss_slope > thresh | rss, rss_slope | Audit allocation, reuse buffers | Linear upward trend
F11 | Duplicate Logic Insert | Similarity > threshold | similarity > 0.9 | similarity, func_id | Reuse existing, annotate difference | N/A
F12 | Unclassified Failure | Detector fallback path | event_type=UNCLASSIFIED | context_hash | Manual triage, add taxonomy entry | Variable
F13 | Inconsistent Telemetry Schema | Schema version mismatch | schema_diff != 0 | schema_version | Run migration, bump version | N/A
F14 | Checkpoint Corruption | Checksum mismatch | checksum_fail | checkpoint_hash | Atomic write, retry | N/A
F15 | Boundary Condition Omission | BC not applied; ghost mismatch | bc_applied=false | face, bc_type | Add BC test, enforce registry | Divergence localized
F16 | Solver Oversolving | Linear solver residual << needed | lin_res_final << tol_req | lin_res_final, iter | Early exit heuristic | Overcompute tail
F17 | Over-Tuning (Non-Transferable Params) | Tuner params degrade other cases | cross_case_penalty > 0 | cross_case_scores | Multi-case objective | Mixed results
F18 | Telemetry Flood | Excess volume causing IO stall | events/sec > limit | events_sec | Sampling throttle | Spiky IO
F19 | Security Policy Violation | Disallowed dep or secret | scanner finding | dep_id/type | Remove/replace dep | N/A
F20 | Reference Drift | Benchmark reference hash changed | ref_hash != stored | ref_hash, prev_hash | Revalidate source, update attestation | N/A

Coverage Goal: 0 unclassified (F12) across benchmark matrix.
