# Algorithm Overview

This project employs a fractional-step (projection) method for incompressible flow:
1. Predict intermediate velocity with explicit advection and diffusion.
2. Solve Poisson equation for pressure correction (Laplacian of pressure equals divergence of predicted velocity).
3. Correct velocity: u <- u* - grad(p).

Advection schemes: Upwind (robust), QUICK (higher-order, more dispersive).

Linear solvers / preconditioners (current):
- Jacobi (baseline)
- ILU (incomplete LU) for improved conditioning
- Multigrid (placeholder scaffold)

Future roadmap: full geometric multigrid, advanced boundary conditions, passive scalar transport, 3D extension.
