# Test Suite Overview

## Active Core Tests (curated)
- `test_config_state_residuals.py`: Config validation, state creation, diffusion residual sanity.
- `test_diffusion_convergence.py`: Steady diffusion manufactured solution convergence (order > ~1.5 currently; target ≥1.9 once SOR/multigrid added).
- `test_poisson_gate.py`: CI-style Poisson manufactured solution gate (order threshold + residual tolerance).

## Disabled / Legacy Suites
Additional test folders (e.g., `pyfoamclone/tests/`, `VS Studio/pyfoamclone/tests/`) exist and are intentionally excluded by `pytest.ini` to avoid noise until refactored or archived.

## Roadmap Additions
- Projection solver tests (once kernels implemented).
- Performance regression tests capturing iteration/time JSON.
- Golden reference comparison (cavity, Poiseuille) harness.

## Running
`pytest` (will default to core tests). Add `-k slow` or marker strategy when slow tests introduced.
