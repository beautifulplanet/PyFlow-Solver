# pyfoamclone

Synthetic CFD scaffold: currently generates benchmark-like centerline velocity profiles (Ghia et al. 1982) for testing infrastructure. It does NOT yet solve the Navier–Stokes equations.

## Status (Aug 2025)
- Placeholder solvers (`PyFOAMSolver`, `FinitudeSolver`) emit deterministic synthetic fields.
- Benchmark data centralized in `pyfoamclone/benchmarks/data.py`.
- Roadmap: implement physical discretization, remove synthetic profiles, add residual tracking, CI, and documentation.

## Install
```bash
pip install -e .[dev]
```

## Run Tests
```bash
pytest -q --cov=pyfoamclone --cov-report=term-missing
```

## Continuous Integration
GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint (ruff), mypy, and tests across Python 3.10–3.13.

## Code Coverage
Install coverage extras:
```bash
pip install coverage pytest-cov
```
Run with coverage (see above). Target threshold (future): 85% lines.

## Disclaimer
Do not use for scientific or engineering decisions yet. Outputs are synthetic and only intended for test harness development.
