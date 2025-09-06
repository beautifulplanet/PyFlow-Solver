# Unit Tests
All Fortran unit tests are now located in the `tests/` directory (not `src/`).

To run all Fortran unit tests (parameters, fields, solver NaN, I/O, BC):
```
make test_all
```
Or run individual tests:
```
make test_parameters
make test_fields
make test_solver_nan
make test_io_utils
make bc_test
```
# HPF-CFD: High-Performance Fortran CFD Solver

## Mission
A modular 2D incompressible lid-driven cavity solver with a modern Fortran core and Python-based workflow & visualization.

## Directory Structure
```
hpf-cfd/
  src/          Fortran source modules (no tests)
  tests/        Fortran unit tests (test_*.f90)
  build/        Objects & executable output
  post/         Python scripts (run & visualize)
  validation/   Benchmark data & validation scripts
  output/       Simulation outputs (results.dat, plots)
  Makefile      Build script
  README.md     Documentation
```

## Build
Requires `gfortran`.

### Windows (Recommended: MSYS2 MINGW64)
1. Install MSYS2: https://www.msys2.org/
2. Update & install toolchain (MINGW64 shell):
  pacman -Syu
  pacman -S --needed base-devel
  pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-make
3. Add `C:\msys64\mingw64\bin` to PATH.
4. Open a new PowerShell in project root.

Then build:

```
make
```
Executable: `build/hpf_cfd.exe`

Clean:
```
make clean
```

## Run
From project root:
```
python post/run_and_plot.py
```
This script will run the solver and produce `output/results.dat` and a PNG plot.

Residual history is written to `output/residuals.dat` every `log_interval` iterations (set in `parameters.f90`).

### Boundary Condition Test
Build & run automated BC test:
```
make bc_test
```
Creates `output/bc_test.dat` and reports PASS/FAIL.

## Next Steps (Roadmap Summary)
1. Flesh out numerical solver (projection method).
2. Add residual monitoring & NaN checks.
3. Implement validation vs Ghia et al. (1982) Re=100.
4. Grid refinement study.
5. Documentation & code comments expansion.

## Parameters
Modify defaults in `src/parameters.f90` (Re, grid size, iterations, etc.).

## License
(Choose a license and add here.)
