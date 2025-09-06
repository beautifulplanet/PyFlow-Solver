# PyFlow CFD Solver Validation

This directory contains tools and tests for validating the accuracy of the PyFlow CFD solver against established benchmark data and performing grid independence studies.

## Validation Tests

### Ghia et al. (1982) Benchmark Comparison

The `test_validation.py` file contains comprehensive validation tests that compare the solver's results with the widely-cited benchmark data from Ghia, Ghia, and Shin (1982). These tests:

1. Run the solver for Re=100, 400, and 1000
2. Compare centerline velocity profiles with benchmark data
3. Calculate error metrics (RMSE)
4. Generate visualization plots of the results

To run the validation tests:

```bash
# Run the basic validation test for Re=100
python -m pytest tests/test_validation.py::test_validation_re100_comparison -v

# Run grid independence study
python -m pytest tests/test_validation.py::test_grid_independence_study -v

# Run all validation tests (may take a long time)
python -m pytest tests/test_validation.py -v
```

### Visualization of Results

You can generate detailed flow visualizations using the provided script:

```bash
# For Re=100 on a 65x65 grid
python scripts/visualize_flow.py --re 100 --grid 65 --time 20.0 --output results

# For Re=400
python scripts/visualize_flow.py --re 400 --grid 65 --time 40.0 --output results

# For Re=1000
python scripts/visualize_flow.py --re 1000 --grid 97 --time 80.0 --output results
```

This will generate:
- Velocity vector fields
- Streamline plots
- Vorticity contours
- Pressure contours
- Centerline velocity profiles
- Residual history

### Grid Independence Study

The grid independence study verifies that your solution is not dependent on the grid resolution. This is a critical validation step in CFD.

```bash
# Run the grid independence study script for Re=100
python scripts/grid_independence_study.py --re 100 --grids 17,33,49,65,97 --time 20.0 --output grid_study

# For Re=400
python scripts/grid_independence_study.py --re 400 --grids 33,49,65,97,129 --time 40.0 --output grid_study
```

This will:
- Run simulations at multiple grid resolutions
- Extract key metrics (vortex position, centerline velocities)
- Calculate observed order of convergence
- Generate convergence plots
- Save a detailed analysis report

## Understanding Convergence

### Order of Convergence

For a numerical method of order p, the error should decrease as O(h^p), where h is the grid spacing. The theoretical order for:
- First-order methods: p ≈ 1
- Second-order methods: p ≈ 2

The observed order in your grid independence study should ideally match the theoretical order of your discretization scheme.

### Richardson Extrapolation

The grid independence study uses Richardson extrapolation to estimate the "exact" solution that would be obtained with an infinitely fine grid. This provides a more accurate reference than simply using the finest grid solution.

### Grid Convergence Index (GCI)

The GCI provides an estimate of the uncertainty in your solution due to the discretization. A lower GCI indicates higher confidence in your numerical results.

## Improving Accuracy

If you find that your solver is not matching benchmark data with sufficient accuracy, or if the order of convergence is lower than expected, consider:

1. **Upgrading Discretization Schemes**: Move from first-order to second-order or higher schemes for the convection terms
2. **Improving Pressure-Velocity Coupling**: Consider SIMPLE, SIMPLEC, or PISO algorithms for better coupling
3. **Refining Grid Near Boundaries**: Use non-uniform grids with clustering near walls
4. **Increasing Pressure Iteration Count**: More iterations in the pressure Poisson equation solver
5. **Reducing Under-relaxation**: If stability isn't an issue, reduce under-relaxation to improve accuracy

## Reference

Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of computational physics, 48(3), 387-411.
