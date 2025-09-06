# CFD Project Status & Red-Team Progress

## Project File Tree

```
[TO BE AUTO-UPDATED: Inserted by script]
```

## Milestone Checklist (Risk-Prioritized)

- [ ] **Red-Team: Randomized field divergence reduction** (core correctness)
- [ ] **Red-Team: Idempotency and divergence-free preservation**
- [ ] **Red-Team: Small/large grid stress**
- [ ] **Red-Team: Energy monotonicity**
- [ ] **Red-Team: Iteration budget enforcement**
- [ ] **Red-Team: Fuzz dt, viscosity, BCs**
- [ ] **Red-Team: Distribution regression**
- [ ] **Red-Team: Performance regression (Jacobi/SOR/MG)**
- [ ] **Red-Team: Adaptive tolerance logic**
- [ ] **Red-Team: Boundary condition edge cases**
- [ ] **Red-Team: Multigrid edge cases**
- [ ] **Red-Team: Logging/telemetry**
- [ ] **Red-Team: Passive scalar module**
- [ ] **Red-Team: Advanced outflow/inflow BCs**
- [ ] **Red-Team: 3D scaffolding**
- [ ] **Red-Team: CI/CD pipeline hardening**
- [ ] **Red-Team: Documentation/roadmap transparency**

## Test Coverage & Red-Team Findings

- [ ] All core projection/Poisson tests pass
- [ ] All hardening/robustness tests pass
- [ ] Performance regression test active
- [ ] Warnings/errors tracked and resolved
- [ ] All new edge cases covered

## Key Architectural Decisions

- Modular solver abstraction (Jacobi/SOR/MG)
- Pluggable boundary conditions
- Performance harness and regression guard
- Adaptive tolerance and logging hooks

## Open Risks / Technical Debt

- [ ] SOR/MG performance on large grids
- [ ] Numerical stability for extreme dt/nu
- [ ] Test coverage for new BCs and 3D
- [ ] Documentation for new contributors

---

*This file is auto-updated after every major change. For full details, see README.md and test reports.*
