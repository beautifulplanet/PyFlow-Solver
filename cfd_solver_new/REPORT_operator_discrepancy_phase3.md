# Phase 3 Diagnostic Report: Operator Discrepancy (Matrix-Free vs Legacy)

Date: 2025-09-05

## 1. Objective
Establish the precise mathematical discrepancies between:
- Legacy assembled negative Laplacian matrix (`assemble_negative_laplacian`)
- Matrix-free composition: `-div(grad(p))` built from existing `gradient` and `divergence` operators

Goal: Explain why the matrix-free projection underperforms and, initially, was unstable.

## 2. Method
A diagnostic tool (`pyflow.tools.operator_discrepancy`) was created to:
1. Assemble legacy matrix A_legacy (with identity boundary rows + reference cell pin).
2. Construct Laplacian L via composition; form `A_mf_neg = -L`.
3. Enforce boundary identity rows to create `A_mf_equiv`.
4. Compute row-wise residual matrices:
   - Raw difference: `D_raw = A_legacy - A_mf_neg`
   - Boundary-aligned difference: `D_equiv = A_legacy - A_mf_equiv`
5. Report per-row norms, classify interior vs boundary, and dump sample stencils.

Grid analyzed: 16 × 16, dx = dy = 1.

## 3. Quantitative Findings
| Category | Interior Max Row Norm | Interior Mean Row Norm | Interior Rows Diff >1e-12 | Boundary Max | Boundary Rows Diff >1e-12 |
|----------|----------------------|------------------------|---------------------------|--------------|---------------------------|
| Legacy - (-L) (raw) | 3.64e+00 | 3.55e+00 | 196 | 2.55e+00 | 60 |
| Legacy - mf_equiv (after boundary identity) | 3.64e+00 | 3.55e+00 | 196 | 0.0 | 0 |

All 196 interior rows (full interior set) differ after boundary alignment.

### Sample Interior Row (Row 17)
Legacy 5-point stencil (flatten index 17):
```
{1: -1.0, 16: -1.0, 17: 4.0, 18: -1.0, 33: -1.0}
```
Matrix-free (-div grad) result:
```
{1: -0.5, 16: -0.5, 17: 1.5, 19: -0.25, 49: -0.25}
```
Differences:
- Center coefficient reduced (1.5 vs 4.0)
- Neighbor coefficients halved or quartered
- Expected immediate neighbors at +x and +y replaced by farther offsets (e.g. 19 instead of 18, 49 instead of 33) → widened stencil shift (two-cell stride)
- Missing direct coupling to some adjacent cells; introduces longer-range couplings.

## 4. Root Cause Analysis
The composed operator uses identical central-difference formulas for both first derivatives:
- grad_x p(i) = (p[i+1] - p[i-1]) / (2Δx)
- div(grad)_x(i) = (grad_x p[i+1] - grad_x p[i-1]) / (2Δx)
Combining yields a second derivative approximation:
```
(p[i+2] - 2p[i] + p[i-2]) / (4 Δx^2)
```
This is a *wider*, lower-magnitude stencil than the classic 3-point second derivative:
```
(p[i+1] - 2p[i] + p[i-1]) / (Δx^2)
```
Because the same pattern applies in y, the resulting Laplacian is *not* the 5-point stencil implemented by the legacy matrix. Instead it is effectively a “dilated” Laplacian with stride-2 coupling and reduced scaling (1/4 factor per directional second derivative), explaining:
- Under-damping of divergence (weaker correction force).
- Initial catastrophic divergence growth when combined with boundary identity assumptions and pressure correction expecting the stronger 5-point operator.

Boundary identity alignment removed boundary discrepancies, confirming the mismatch is purely interior operator construction — not a sign flip issue.

## 5. Physics-Based Hypothesis
The projection relies on solving (−∇² p = RHS) with a discretization consistent with the velocity divergence operator. The legacy system implicitly assumes a 5-point Laplacian consistent with a *direct* second derivative scheme. The matrix-free composition, however, produces a different discrete operator that:
- Applies a longer-wavelength smoothing (stride-2 second derivative) and
- Reduces corrective pressure curvature by roughly a factor of ~2–2.5 in magnitude at the center.
Thus, pressure gradients derived from this p fail to remove sufficient divergence in a single pass, preventing the required ≥100× reduction. Instability episodes stem from mismatch between expected and actual spectral damping properties.

## 6. Conclusion
The matrix-free operator as implemented (div(grad)) is **not mathematically equivalent** to the assembled legacy negative Laplacian. Its use in projection without redesign of the discrete gradient/divergence pair or explicit adjustment to a proper second-derivative formulation is invalid under current solver assumptions.

## 7. Remediation Plan (Proposed)
Ranked by rigor + minimal disruption:
1. Implement direct 5-point matrix-free negative Laplacian apply (`apply_neg_laplacian_5pt`) mirroring legacy coefficients (interior only), leaving boundary handling to solver (identity rows / pin logic as today). Use this in the matrix-free path instead of composing div(grad).
2. Add unit test: compare `apply_neg_laplacian_5pt(e_k)` vs assembled matrix column (tolerance ~1e-14) for several interior indices.
3. Replace current LinearOperator implementation with the new direct stencil.
4. Re-run operator discrepancy tool (interior differences should drop to ~0 within tolerance).
5. Re-run gatekeeper divergence tests; expect stronger correction. If still <100×, investigate velocity correction sign or time-step scaling.
6. Only after passing: remove legacy non-physical divergence “hotfix.”
7. (Optional) Longer term: If true composition-based Laplacian is desired, redesign discrete operators to form a consistent collocated second derivative (e.g., use forward/backward then central) or shift to a staggered layout.

## 8. Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Incorrect stencil replication | Continued failure to reduce divergence | Column-by-column verification tests |
| Hidden dependency on boundary identity rows | Non-physical pressure near edges | Introduce proper Neumann / reference cell handling later |
| Spectral mismatch after fix | Sub-100× reduction persists | Analyze eigenvalues (FFT on periodic subgrid) |

## 9. Authorization Needed
Authorization requested to proceed with Step 1–3 of remediation. No further projection integration changes will occur until interior equality with legacy matrix is verified.

---
Prepared for Strategos & N-V.A.L.
