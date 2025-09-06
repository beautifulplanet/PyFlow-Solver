# Recommended CFD Scaffold

Core functional areas detected: velocity, pressure, boundary, solver, residual, flux, mesh, cfd

Suggested module layout:

- cfd_core/__init__.py
- cfd_core/mesh.py  # mesh generation & boundary tagging
- cfd_core/physics.py  # flux computations, material properties
- cfd_core/solver.py  # time-stepping orchestration
- cfd_core/boundary_conditions.py
- cfd_core/postprocess.py
- experiments/    # prototype and research scripts
- scripts/run_case.py
- tests/test_solver.py
- tests/test_fluxes.py

Implementation guidance:
- Keep functions < 60 lines; refactor when complexity > 12.
- Centralize external imports; aim to reduce rarely reused ones.
- Record experiment metadata (params, hash) to reproducibility log.