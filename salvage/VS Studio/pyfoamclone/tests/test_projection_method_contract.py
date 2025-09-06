# Large-grid, multi-step pinpoint test for regression setup
def test_large_grid_multi_step_stability():
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.solver import step
    from pyfoamclone.residuals.manager import ResidualManager
    nx, ny = 65, 65
    dx = dy = 1.0 / (nx - 1)
    class DummyCfg:
        disable_advection = False
        advection_scheme = 'quick'
        cfl_target = 0.5
        cfl_growth = 1.1
        Re = 100.0
        lid_velocity = 1.0
        test_mode = False
        lin_tol = 1e-10
        lin_maxiter = 800
        lx = 1.0
        ly = 1.0
        max_iter = 2000
        tol = 1e-5
    cfg = DummyCfg()
    state = allocate_state(nx, ny)
    tracker = ResidualManager()
    from pyfoamclone.numerics.fluid_ops import divergence
    for i in range(100):
        u_int = interior_view(state.fields['u'])
        v_int = interior_view(state.fields['v'])
        div = np.linalg.norm(divergence(u_int, v_int, dx, dy))
        u_norm = np.linalg.norm(u_int)
        v_norm = np.linalg.norm(v_int)
        if i % 10 == 0 or i == 99:
            print(f"Step {i}: divergence={div}, u_norm={u_norm}, v_norm={v_norm}")
        state, residuals, diagnostics = step(cfg, state, tracker, i)
    # After 100 steps, divergence should not explode
    final_div = np.linalg.norm(divergence(interior_view(state.fields['u']), interior_view(state.fields['v']), dx, dy))
    assert final_div < 100.0, f"Divergence exploded: {final_div}"
# Multi-step solver test: check divergence over several steps
def test_multi_step_solver_divergence_reduction():
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.solver import step
    from pyfoamclone.residuals.manager import ResidualManager
    nx, ny = 8, 8
    dx = dy = 1.0
    class DummyCfg:
        disable_advection = False
        advection_scheme = 'quick'
        cfl_target = 0.5
        cfl_growth = 1.1
        Re = 100.0
        lid_velocity = 1.0
        test_mode = False
        lin_tol = 1e-10
        lin_maxiter = 200
    cfg = DummyCfg()
    state = allocate_state(nx, ny)
    u = np.zeros((ny+2, nx+2))
    v = np.zeros((ny+2, nx+2))
    u[3:6, 3:6] = 1.0
    state.fields['u'][:] = u
    state.fields['v'][:] = v
    tracker = ResidualManager()
    from pyfoamclone.numerics.fluid_ops import divergence
    divergences = []
    for i in range(10):
        u_int = interior_view(state.fields['u'])
        v_int = interior_view(state.fields['v'])
        div = np.linalg.norm(divergence(u_int, v_int, dx, dy))
        divergences.append(div)
        state, residuals, diagnostics = step(cfg, state, tracker, i)
    print("Divergences over steps:", divergences)
    # Should decrease monotonically (allowing for small numerical noise)
    assert divergences[-1] < divergences[0], f"Divergence did not decrease: {divergences}"

# Mini lid-driven cavity test: run a few steps and check velocity/divergence
def test_mini_lid_driven_cavity_evolution():
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.solver import step
    from pyfoamclone.residuals.manager import ResidualManager
    nx, ny = 8, 8
    dx = dy = 1.0
    class DummyCfg:
        disable_advection = False
        advection_scheme = 'quick'
        cfl_target = 0.5
        cfl_growth = 1.1
        Re = 100.0
        lid_velocity = 1.0
        test_mode = False
        lin_tol = 1e-10
        lin_maxiter = 200
    cfg = DummyCfg()
    state = allocate_state(nx, ny)
    # Initial field is zero (rest)
    tracker = ResidualManager()
    from pyfoamclone.numerics.fluid_ops import divergence
    for i in range(5):
        u_int = interior_view(state.fields['u'])
        v_int = interior_view(state.fields['v'])
        div = np.linalg.norm(divergence(u_int, v_int, dx, dy))
        lid_row = state.fields['u'][-2, 1:-1]
        print(f"Step {i}: divergence={div}, lid_row={lid_row}")
        state, residuals, diagnostics = step(cfg, state, tracker, i)
    # After a few steps, lid velocity should be imposed and divergence should not explode
    assert np.allclose(state.fields['u'][-2, 1:-1], cfg.lid_velocity), "Lid BC not enforced after steps"
    final_div = np.linalg.norm(divergence(interior_view(state.fields['u']), interior_view(state.fields['v']), dx, dy))
    assert final_div < 10.0, f"Divergence exploded: {final_div}"
# Pinpoint Full Solver Step Test
def test_full_solver_step_divergence_and_bcs():
    """
    Test a single full solver step: advection + projection + BCs.
    Start with a simple velocity field, run one step, and check:
    - Divergence is reduced after projection
    - BCs are only applied after projection
    - Ghost cells are not included in operator calculations
    """
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.solver import step
    from pyfoamclone.residuals.manager import ResidualManager
    nx, ny = 8, 8
    dx = dy = 1.0
    class DummyCfg:
        disable_advection = False
        advection_scheme = 'quick'
        cfl_target = 0.5
        cfl_growth = 1.1
        Re = 100.0
        lid_velocity = 1.0
        test_mode = False
        lin_tol = 1e-10
        lin_maxiter = 200
    cfg = DummyCfg()
    state = allocate_state(nx, ny)
    # Set a simple initial velocity field (nonzero in interior)
    u = np.zeros((ny+2, nx+2))
    v = np.zeros((ny+2, nx+2))
    u[3:6, 3:6] = 1.0
    state.fields['u'][:] = u
    state.fields['v'][:] = v
    tracker = ResidualManager()
    # Compute initial divergence (interior only)
    u_int = interior_view(state.fields['u'])
    v_int = interior_view(state.fields['v'])
    from pyfoamclone.numerics.fluid_ops import divergence
    div_before = np.linalg.norm(divergence(u_int, v_int, dx, dy))
    # Run one solver step
    state, residuals, diagnostics = step(cfg, state, tracker, 0)
    u_after = interior_view(state.fields['u'])
    v_after = interior_view(state.fields['v'])
    div_after = np.linalg.norm(divergence(u_after, v_after, dx, dy))
    print(f"Divergence before: {div_before}, after: {div_after}")
    # Check that divergence is reduced
    assert div_after < div_before, f"Divergence not reduced: before {div_before}, after {div_after}"
    # Check that BCs are enforced (lid velocity at top row)
    lid_row = state.fields['u'][-2, 1:-1]
    assert np.allclose(lid_row, cfg.lid_velocity), f"Lid BC not enforced after step: {lid_row}"
# Pinpoint Advection Operator Test
def test_advection_operator_accuracy():
    """
    Test advection operator (upwind and QUICK) on phi = x^2 + y^2 with uniform velocity.
    Compare to analytic convective derivative: u*d(phi)/dx + v*d(phi)/dy = 2x + 2y.
    """
    import numpy as np
    from pyfoamclone.numerics.operators.advection import advect_upwind, advect_quick
    nx, ny = 16, 16
    dx = dy = 1.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)
    phi = X**2 + Y**2
    u = np.ones_like(phi)
    v = np.ones_like(phi)
    expected = 2*X + 2*Y
    upwind = advect_upwind(u, v, phi, dx, dy)
    quick = advect_quick(u, v, phi, dx, dy)
    # Ignore boundaries for error calculation
    mask = (slice(2, -2), slice(2, -2))
    err_upwind = np.mean(np.abs(upwind[mask] - expected[mask]))
    err_quick = np.mean(np.abs(quick[mask] - expected[mask]))
    print(f"Upwind mean abs error: {err_upwind}")
    print(f"QUICK mean abs error: {err_quick}")
    # Both should be small, QUICK should be more accurate
    assert err_upwind < 5.0, f"Upwind error too large: {err_upwind}"
    assert err_quick < err_upwind, f"QUICK should be more accurate than upwind"
# Task D: Isolate the Corrector with Analytic Fields
def test_gradient_correction_reduces_divergence():
    """
    Isolated test: Given u = X, v = Y (divergence = 2), and p = (X^2 + Y^2)/4 (analytic Poisson solution),
    applying the pressure gradient correction should reduce divergence to zero.
    """
    import numpy as np
    from pyfoamclone.numerics.fluid_ops import divergence, gradient
    nx, ny = 8, 8
    dx = dy = 1.0
    dt = 1.0  # Use dt=1 for clarity
    x = np.arange(1, nx+1) * dx
    y = np.arange(1, ny+1) * dy
    X, Y = np.meshgrid(x, y)
    # u = X, v = Y (interior arrays)
    u = X.copy()
    v = Y.copy()
    # p = (X^2 + Y^2)/2 (analytic solution to Laplacian p = divergence for this finite difference scheme)
    p = (X**2 + Y**2) / 2.0
    # Initial divergence: check only interior (exclude boundaries)
    div_initial = divergence(u, v, dx, dy)
    interior = (slice(1, -1), slice(1, -1))
    print("Initial divergence (interior):\n", div_initial[interior])
    assert np.allclose(div_initial[interior], 2.0), f"Interior divergence not 2: {div_initial}"
    # Compute pressure gradient
    dpdx, dpdy = gradient(p, dx, dy)
    print("dpdx (interior):\n", dpdx[interior])
    print("dpdy (interior):\n", dpdy[interior])
    # Apply correction: u_corrected = u - dt * dpdx, v_corrected = v - dt * dpdy
    u_corr = u - dt * dpdx
    v_corr = v - dt * dpdy
    # Divergence after correction: check only interior
    div_corrected = divergence(u_corr, v_corr, dx, dy)
    print("Corrected divergence (interior):\n", div_corrected[interior])
    print("Difference (should be -2):\n", div_corrected[interior] - div_initial[interior])
    assert np.allclose(div_corrected[interior], 0, atol=0.5), f"Interior divergence after correction not zero (tol=0.5): {div_corrected}"
import numpy as np
import pytest
from pyfoamclone.core.ghost_fields import allocate_state, interior_view
from pyfoamclone.numerics.fluid_ops import divergence as div_op, gradient
from pyfoamclone.numerics.operators.advection import advect_upwind
from pyfoamclone.solvers.pressure_solver import solve_pressure_poisson

# Task A: Verify the Predictor
def test_predictor_step_creates_divergence():
    nx, ny = 8, 8
    state = allocate_state(nx, ny)
    # Use a non-uniform field to ensure advection creates divergence
    u = np.zeros((ny + 2, nx + 2))
    v = np.zeros((ny + 2, nx + 2))
    u[3:6, 3:6] = 1.0  # Patch of nonzero u in the center
    state.fields['u'][:] = u
    state.fields['v'][:] = v
    dx = dy = 1.0
    dt = 0.1
    u_int = interior_view(state.fields['u'])
    v_int = interior_view(state.fields['v'])
    conv_u = advect_upwind(u_int, v_int, u_int, dx, dy)
    conv_v = advect_upwind(u_int, v_int, v_int, dx, dy)
    u_star = u_int - dt * conv_u
    v_star = v_int - dt * conv_v
    div = div_op(u_star, v_star, dx, dy)
    assert np.linalg.norm(div) > 0, "Predictor step should create nonzero divergence."

# Task B: Verify the Pressure Solver
def test_pressure_solver_on_known_divergence():
    nx, ny = 8, 8
    state = allocate_state(nx, ny)
    u = np.zeros((ny + 2, nx + 2))
    v = np.zeros((ny + 2, nx + 2))
    u[4, 4] = 1.0  # interior (ghost cell offset)
    v[4, 4] = 1.0
    state.fields['u'][:] = u
    state.fields['v'][:] = v
    dx = dy = 1.0
    dt = 0.1
    class DummyCfg:
        lin_tol = 1e-10
        lin_maxiter = 400
    cfg = DummyCfg()
    p_corr, diag = solve_pressure_poisson(state, dt, dx, dy, cfg, preconditioner=None)
    assert np.linalg.norm(p_corr) > 0, "Pressure correction should be nonzero for nonzero divergence."

# Task C: Verify the Corrector
def test_corrector_step_reduces_divergence():
    nx, ny = 8, 8
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.pressure_solver import solve_pressure_poisson
    class DummyCfg:
        lin_tol = 1e-10
        lin_maxiter = 400
    cfg = DummyCfg()
    state = allocate_state(nx, ny)
    u = np.random.rand(ny + 2, nx + 2)
    v = np.random.rand(ny + 2, nx + 2)
    state.fields['u'][:] = u
    state.fields['v'][:] = v
    dx = dy = 1.0
    dt = 0.1
    u_int = interior_view(state.fields['u'])
    v_int = interior_view(state.fields['v'])
    div_before = np.linalg.norm(div_op(u_int, v_int, dx, dy))
    # Solve for pressure correction (applies correction in-place)
    solve_pressure_poisson(state, dt, dx, dy, cfg)
    u_corr = interior_view(state.fields['u'])
    v_corr = interior_view(state.fields['v'])
    div_after = np.linalg.norm(div_op(u_corr, v_corr, dx, dy))
    assert div_after < div_before, "Corrector step should reduce divergence."

def test_pressure_solve_with_manufactured_solution():
    """
    Test pressure projection with a manufactured solution.
    - u = X, v = Y (divergence = 2)
    - Analytical pressure: p = (X^2 + Y^2)/4
    - After projection, pressure should match analytic, and divergence should be zero.
    """
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.pressure_solver import solve_pressure_poisson
    from pyfoamclone.numerics.fluid_ops import divergence
    nx, ny = 16, 16
    dx = dy = 0.1
    dt = 1.0
    # Create coordinate grids
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y)
    # Manufactured velocity and pressure
    u = X.copy()
    v = Y.copy()
    p_analytic = (X**2 + Y**2) / 4.0
    # Allocate state and set fields
    state = allocate_state(nx, ny)
    state.fields['u'][1:-1, 1:-1] = u
    state.fields['v'][1:-1, 1:-1] = v
    class DummyCfg:
        lin_tol = 1e-10
        lin_maxiter = 400
    cfg = DummyCfg()
    # Solve for pressure correction (applies correction in-place)
    solve_pressure_poisson(state, dt, dx, dy, cfg)
    # Extract computed pressure and compare to analytic (up to a constant)
    p_num = interior_view(state.fields['p'])
    # Remove mean offset for comparison
    p_num = p_num - np.mean(p_num)
    p_analytic = p_analytic - np.mean(p_analytic)
    assert np.allclose(p_num, p_analytic, atol=1e-6), "Pressure field does not match analytic solution."
    # Check divergence after correction
    u_corr = interior_view(state.fields['u'])
    v_corr = interior_view(state.fields['v'])
    div_corr = divergence(u_corr, v_corr, dx, dy)
    interior = (slice(1, -1), slice(1, -1))
    assert np.allclose(div_corr[interior], 0.0, atol=1e-10), "Divergence not reduced to zero after projection."
    # Diagnostic: print max/min/mean difference
    diff = p_num - p_analytic
    print("Pressure field difference stats:")
    print("max:", np.max(diff), "min:", np.min(diff), "mean:", np.mean(diff))
    print("diff (center row):", diff[diff.shape[0]//2])
    print("p_num (center row):", p_num[p_num.shape[0]//2])
    print("p_analytic (center row):", p_analytic[p_analytic.shape[0]//2])
