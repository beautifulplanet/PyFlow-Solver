# Projection / Poisson Solver Notes

High‑level: Fractional step (unscaled Chorin). Solve Lap(p)=div(U*) then correct U.

Environment flags:
- PROJECTION_ENABLE=1 : required to run projection_step.
- PROJECTION_LINSOLVER=jacobi|sor|mg : choose linear solver.
- PROJECTION_SOR_OMEGA (float, default 1.7) : SOR relaxation factor.
- PROJECTION_MG_PRE / PROJECTION_MG_POST : multigrid smoothing sweeps.
- PROJECTION_MG_SMOOTHER=jacobi|wjacobi ; PROJECTION_MG_JACOBI_OMEGA (weighted Jacobi factor).
- PROJECTION_POISSON_BASE_TOL : base residual L_inf tolerance (default 1e-6).
- PROJECTION_ADAPTIVE_TOL=1 : enable adaptive tolerance scaling by divergence.
- PROJECTION_ADAPT_REF : reference divergence scale for adaptive tol (default 1e-3).
- PROJECTION_DEBUG=1 : print per-cycle / per-iteration diagnostics.
- PROJECTION_ASSERT_REDUCTION=1 : raise if divergence not reduced.

Diagnostics:
- Returned ProjectionStats.notes contains div_linf_before, div_linf, poisson_tol, method, residual estimate, etc.
- PROJECTION_BACKEND (exported from cfd_solver.pyflow.core) reveals which module provided projection_step.

Safety:
- dt is CFL-limited (advection + diffusion) and clamped to >=1e-12.
- Optional assertion prevents silent failures in CI.

Testing additions:
- Multigrid speed vs Jacobi (existing test).
- New SOR speed test (`test_poisson_sor.py`).
- Adaptive tolerance scaling test ensures tolerance loosens only when divergence large.

Next potential hardening ideas:
- Add convergence rate regression thresholds for MG and SOR.
- Randomized omega sweep to detect performance regressions.
- Benchmark script to record wall-clock vs iteration counts.