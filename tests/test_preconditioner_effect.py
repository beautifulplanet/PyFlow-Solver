import numpy as np
from types import SimpleNamespace
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.solvers.pressure_solver import solve_pressure_poisson

def _baseline_cfg(nx, ny):
    return SimpleNamespace(nx=nx, ny=ny, lin_tol=1e-10, lin_maxiter=300, diagnostics=False, enable_jacobi_pc=True)

def test_jacobi_preconditioner_reduces_iterations_preserves_solution():
    nx, ny = 24, 20
    cfg_pc = _baseline_cfg(nx, ny)
    cfg_nopc = _baseline_cfg(nx, ny)
    setattr(cfg_nopc, 'enable_jacobi_pc', False)
    state_pc = allocate_state(nx, ny)
    state_nopc = allocate_state(nx, ny)
    # Seed identical initial fields (zeros already) – create synthetic provisional divergence by putting a bump in u
    interior_view(state_pc.fields['u'])[ny//3, nx//3] = 1.0
    interior_view(state_nopc.fields['u'])[ny//3, nx//3] = 1.0
    dt = 0.01; dx = 1.0; dy = 1.0
    p_pc, diag_pc = solve_pressure_poisson(state_pc, dt, dx, dy, cfg_pc)
    p_nopc, diag_nopc = solve_pressure_poisson(state_nopc, dt, dx, dy, cfg_nopc)
    # Compare pressure fields (up to tolerance)
    assert np.allclose(p_pc, p_nopc, rtol=1e-10, atol=1e-10)
    it_pc = diag_pc['Rp_iterations']
    it_nopc = diag_nopc['Rp_iterations']
    # Preconditioned should not take MORE iterations; usually strictly fewer
    # Require strictly fewer iterations unless baseline already trivial (<=2)
    if it_nopc > 2:
        assert it_pc < it_nopc, f"Expected preconditioned iterations < unpreconditioned ({it_pc} vs {it_nopc})"
    else:
        assert it_pc <= it_nopc
    # Sanity: solutions not all zeros (divergence was non-zero)
    assert np.linalg.norm(p_pc) > 0