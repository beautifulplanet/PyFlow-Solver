def test_advection_of_scalar_blob():
    """
    Pure advection test: diagonal velocity, Gaussian blob, nu=0.
    The blob should move diagonally by (u*dt*steps, v*dt*steps).
    """
    import numpy as np
    from pyfoamclone.core.ghost_fields import allocate_state, interior_view
    from pyfoamclone.solvers.solver import step
    nx, ny = 32, 32
    dx, dy = 1.0, 1.0
    dt = 0.5
    steps = 8
    u0, v0 = 1.0, 1.0
    # Set up state
    state = allocate_state(nx, ny)
    u = interior_view(state.fields['u'])
    v = interior_view(state.fields['v'])
    u[:,:] = u0
    v[:,:] = v0
    # Gaussian blob in lower left
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dy)
    blob = np.exp(-((X-4)**2 + (Y-4)**2)/4.0)
    state.fields['phi'] = np.pad(blob, ((1,1),(1,1)), mode='constant')
    # Disable diffusion and projection
    class DummyConfig:
        disable_advection = False
        pass
    cfg = DummyConfig()
    cfg.nx = nx
    cfg.ny = ny
    cfg.lx = dx*(nx-1)
    cfg.ly = dy*(ny-1)
    cfg.Re = 1e12  # effectively nu=0
    cfg.max_iter = 1
    cfg.cfl_target = 1.0
    cfg.cfl_growth = 1.0
    cfg.tol = 1e-12
    for _ in range(steps):
        # Only advect phi, not u/v
        phi = interior_view(state.fields['phi'])
        from pyfoamclone.numerics.operators.advection import advect_upwind
        phi_new = phi - dt * advect_upwind(u, v, phi, dx, dy)
        state.fields['phi'][1:-1,1:-1] = phi_new
    # The blob should have moved by (u0*dt*steps, v0*dt*steps)
    shift = int(round(u0*dt*steps))
    blob_final = np.exp(-((X-4-shift)**2 + (Y-4-shift)**2)/4.0)
    phi_end = interior_view(state.fields['phi'])
    # Compare max location
    max_idx = np.unravel_index(np.argmax(phi_end), phi_end.shape)
    max_idx_ref = np.unravel_index(np.argmax(blob_final), blob_final.shape)
    assert abs(max_idx[0] - max_idx_ref[0]) <= 1 and abs(max_idx[1] - max_idx_ref[1]) <= 1

def test_lid_driven_cavity_Re100_quick_scheme():
    """
    High-accuracy regression: lid-driven cavity, QUICK scheme, 65x65 grid, strict MSE tolerance.
    """
    import numpy as np
    from pathlib import Path
    from pyfoamclone.scripts.run_case import run
    import json
    # Load case and run
    case = Path(__file__).parent.parent / 'cases' / 'lid_cavity_Re100_quick.json'
    result = run(str(case))
    u_centerline = np.array(result['u_centerline'])
    y_centerline = np.array(result['y_centerline'])
    # Load reference
    ref_path = Path(__file__).parent.parent / 'pyfoamclone' / 'benchmarks' / 'ghia_centerline_u_re100.json'
    data = json.loads(ref_path.read_text())
    y_ref_desc = np.array(data['y_desc'])
    u_ref_desc = np.array(data['u_centerline'])
    y_ref_asc = y_ref_desc[::-1]
    u_ref_asc = u_ref_desc[::-1]
    u_sim_interp_asc = np.interp(y_ref_asc, y_centerline, u_centerline)
    u_sim_interp_desc = u_sim_interp_asc[::-1]
    mse = np.mean((u_sim_interp_desc - u_ref_desc)**2)
    assert mse < 5e-5, f"QUICK MSE too high: {mse}"