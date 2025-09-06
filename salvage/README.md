# CFD Mini Framework

Minimal 2D incompressible CFD sandbox evolving toward validated physics with strong test gating.

## Components
- `framework/state.py` mesh & solver state container
- `framework/projection_solver.py` fractional-step projection (Jacobi / SOR)
- `framework/boundaries.py` pluggable boundary condition helpers (no-slip, free-slip, periodic-x, lid-driven cavity)
- `framework/steady_diffusion.py` manufactured diffusion / diffusion kernel
- `poisson_manufactured_test.py` manufactured Poisson convergence + perf metrics (CI gate)
- Tests: diffusion convergence, Poisson performance & gate, projection divergence reduction + red-team hardening (energy, fuzz, distribution, iteration budget), boundary qualitative (lid-driven cavity), linear solver selection (SOR faster than Jacobi)

## Quick Start
```powershell
# (Optional) create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

Run manufactured Poisson gate:
```powershell
python poisson_manufactured_test.py --ci-gate --grids 17 33 65 --tol 1e-6 --order-threshold 1.6
```

Enable projection step test:
```powershell
$env:PROJECTION_ENABLE="1"
pytest tests/test_projection_solver.py::test_projection_reduces_divergence -q
```

Run red-team / hardening suite (slow marked tests included):
```powershell
pytest -q -m slow
```

Select SOR linear solver:
```powershell
$env:PROJECTION_LINSOLVER="sor"; $env:PROJECTION_SOR_OMEGA="1.7"
pytest tests/test_poisson_solver_selection.py::test_sor_not_slower_than_jacobi -q
```

Lid-driven cavity qualitative smoke test:
```powershell
pytest tests/test_lid_driven_cavity_re100.py::test_lid_driven_cavity_basic_structure -q
```

## Environment Flags
| Variable | Purpose | Default |
|----------|---------|---------|
| `PROJECTION_ENABLE` | Enable projection solver | 0 |
| `PROJECTION_LINSOLVER` | `jacobi` or `sor` | jacobi |
| `PROJECTION_SOR_OMEGA` | Relaxation factor for SOR | 1.7 |
| `PROJECTION_DEBUG` | Verbose residual prints | unset |

Adaptive / tuning environment variables:
- `PROJECTION_POISSON_BASE_TOL` (default 1e-6)
- `PROJECTION_ADAPT_REF` (reference divergence magnitude, default 1e-3)
- `PROJECTION_ADAPTIVE_TOL=1` enables scaling tolerance = `base_tol` * max(1, `div_before`/`adapt_ref`) capped at 1e-3
- `PROJECTION_MG_PRE` / `PROJECTION_MG_POST` (int pre/post smoothing sweeps, default 2)
- `PROJECTION_MG_SMOOTHER=jacobi|wjacobi` (weighted jacobi uses `PROJECTION_MG_JACOBI_OMEGA`, default 0.8)
- `PROJECTION_MG_JACOBI_OMEGA` (weight for weighted Jacobi)

## Hardening Tests
- Random field divergence reduction
- Idempotency (second projection does not degrade)
- Divergence-free preservation
- Energy non-inflation
- dt fuzz invariance
- Distribution regression (median & worst reduction ratios)
- Larger grid stress
- Iteration budget & SOR speed advantage
- Lid-driven cavity qualitative structure

## Performance Baseline
Create baseline (writes JSON):
```powershell
$env:PROJECTION_ENABLE="1"
python scripts/perf_projection.py --baseline perf_baselines/projection_perf_baseline.json
```
Compare to baseline:
```powershell
$env:PROJECTION_ENABLE="1"
python scripts/perf_projection.py --compare perf_baselines/projection_perf_baseline.json
```
Slow perf regression test auto-skips if baseline missing: `tests/test_projection_perf_regression.py`.

## Tooling & CI
- Ruff for linting (`ruff check .`)
- Mypy strict typing for `framework/`
- GitHub Actions workflow (`.github/workflows/ci.yml`) running lint, type, fast tests on 3.11 & 3.12; slow tests sampled on 3.11.

## Roadmap
Status legend: [DONE], [WIP], [TODO], [FUTURE]
1. Diffusion convergence & manufactured Poisson [DONE]
2. Baseline projection & divergence reduction [DONE]
3. Hardening fuzz & energy suite [DONE]
4. Linear solver abstraction + SOR [DONE]
5. Boundary condition module [DONE]
6. Lid-driven cavity qualitative test [DONE]
7. Performance regression harness (timing JSON + thresholds) [TODO]
8. CI (GitHub Actions) + lint (ruff) + type (mypy) [TODO]
9. Multigrid V-cycle prototype [FUTURE]
10. Passive scalar transport & coupling [FUTURE]
11. Extended BC (in/outflow, pressure outlet) [FUTURE]
12. Logging & telemetry dashboards [FUTURE]
13. 3D pathfinder (data layout + single step) [FUTURE]

## License
MIT

Supported Poisson solvers (set env PROJECTION_LINSOLVER):
- jacobi (default)
- sor (Successive Over-Relaxation, faster)
- mg (Multigrid V-cycle; typically fastest for larger grids)

To update performance baseline including multigrid:
python scripts/update_perf_baseline.py  # generates perf_baseline.json
