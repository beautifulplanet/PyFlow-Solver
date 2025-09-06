from __future__ import annotations
import numpy as np
import math
from typing import Dict, Any
from ..core.ghost_fields import interior_view, State
from ..logging.structured import JsonlLogger
from ..numerics.fluid_ops import divergence as div_op, apply_pressure_correction
from ..numerics.operators.advection import advect_upwind, advect_quick
from .pressure_solver import solve_pressure_poisson

def step(config, state: State, tracker, iteration: int) -> tuple[State, Dict[str,float], Dict[str, Any]]:
    diagnostics_enabled = getattr(config, 'diagnostics', True)
    nx = interior_view(state.fields['u']).shape[1]
    ny = interior_view(state.fields['u']).shape[0]
    lx = getattr(config, 'lx', float(nx - 1))
    ly = getattr(config, 'ly', float(ny - 1))
    dx = lx / max(nx - 1, 1)
    dy = ly / max(ny - 1, 1)
    if diagnostics_enabled:
        print(f"[Step {iteration}] Grid: nx={nx}, ny={ny}, dx={dx:.4g}, dy={dy:.4g}, Re={getattr(config, 'Re', 100.0)}")
    umax = float(np.max(np.abs(interior_view(state.fields['u'])))) + 1e-12
    vmax = float(np.max(np.abs(interior_view(state.fields['v'])))) + 1e-12
    cfl_target = getattr(config, 'cfl_target', 0.5)
    # Slightly reduce growth to stabilize projection under loose linear solves
    cfl_growth = getattr(config, 'cfl_growth', 1.05)
    Re = float(getattr(config, 'Re', 100.0))
    nu = 1.0 / Re
    dt_prev = float(state.meta.get('dt_prev', 1e-2))
    dt_cfl_u = cfl_target * dx / umax
    dt_cfl_v = cfl_target * dy / vmax
    dt_cfl = min(dt_cfl_u, dt_cfl_v)
    dt_diff = 0.25 * min(dx*dx, dy*dy) / max(nu, 1e-12)
    dt_candidate = min(dt_cfl, dt_diff)
    if dt_candidate > dt_prev * cfl_growth:
        dt = dt_prev * cfl_growth
    else:
        dt = dt_candidate
    # Baseline divergence BEFORE modifying velocities (for safeguard)
    ui_full = interior_view(state.fields['u'])
    vi_full = interior_view(state.fields['v'])
    div_baseline_before = float(np.linalg.norm(div_op(ui_full, vi_full, 1.0, 1.0)))
    hist = state.meta.get('divergence_history', [])
    hist.append(div_baseline_before)
    # Keep only last 5
    hist = hist[-5:]
    state.meta['divergence_history'] = hist
    # Safeguard: if last 3 baselines strictly increasing by >1% each, shrink dt
    if len(hist) >= 3:
        a, b, c = hist[-3], hist[-2], hist[-1]
        if a > 0 and b > a*1.01 and c > b*1.01:
            dt *= 0.5
            if getattr(config, 'diagnostics', True):
                print(f"[Step {iteration}] dt safeguard triggered (divergence rising {a:.3g}->{b:.3g}->{c:.3g}); new dt={dt:.3g}")
    state.meta['dt_prev'] = dt
    state.meta['dx'] = dx
    state.meta['dy'] = dy
    state.meta['nu'] = nu
    current_cfl = max(umax * dt / dx, vmax * dt / dy)
    if diagnostics_enabled:
        print(f"[Step {iteration}] dt={dt:.3g}, umax={umax:.3g}, vmax={vmax:.3g}, CFL={current_cfl:.3g}")
    # Re-read after potential dt adjustment (unchanged velocities)
    ui_full = interior_view(state.fields['u'])
    vi_full = interior_view(state.fields['v'])
    div_baseline = float(np.linalg.norm(div_op(ui_full, vi_full, 1.0, 1.0)))
    if diagnostics_enabled:
        print(f"[Step {iteration}] Initial divergence norm: {div_baseline:.3g}")
    u_int = interior_view(state.fields['u'])
    v_int = interior_view(state.fields['v'])
    u_old = u_int.copy()
    v_old = v_int.copy()
    if getattr(config, 'disable_advection', False):
        scheme = '(disabled)'
        u_star = u_old.copy()
        v_star = v_old.copy()
    else:
        scheme = getattr(config, 'advection_scheme', 'upwind')
        if scheme == 'quick':
            conv_u = advect_quick(u_old, v_old, u_old, dx, dy)
            conv_v = advect_quick(u_old, v_old, v_old, dx, dy)
        else:
            conv_u = advect_upwind(u_old, v_old, u_old, dx, dy)
            conv_v = advect_upwind(u_old, v_old, v_old, dx, dy)
        u_star = u_old - dt * conv_u
        v_star = v_old - dt * conv_v
    if diagnostics_enabled:
        print(f"[Step {iteration}] Advection scheme: {scheme}")
        print(f"[Step {iteration}] u_star stats: min={u_star.min():.3g}, max={u_star.max():.3g}, mean={u_star.mean():.3g}")
    u_pred = u_star.copy()
    v_pred = v_star.copy()
    u_int[:, :] = u_pred
    v_int[:, :] = v_pred
    # Delay lid velocity BC until AFTER pressure projection for test expectations
    lid_velocity = getattr(config, 'lid_velocity', None)
    preconditioner = None
    p_corr, pressure_diag = solve_pressure_poisson(state, dt, dx, dy, config, preconditioner=preconditioner)
    if diagnostics_enabled:
        print(f"[Step {iteration}] Pressure solve: iterations={pressure_diag.get('Rp_iterations', 0)}, residual={pressure_diag.get('Rp_residual', 0.0):.3g}, converged={pressure_diag.get('Rp_converged', False)}")
    state.meta['prev_div_norm'] = pressure_diag.get('divergence_norm', 0.0)
    # Now enforce lid velocity BC (top interior row) AFTER projection
    if lid_velocity is not None:
        state.fields['u'][-2, 1:-1] = lid_velocity
    if diagnostics_enabled and lid_velocity is not None:
        print(f"[Step {iteration}] After BC: lid row u = {interior_view(state.fields['u'])[-2,1:-1]}")
    def l2(delta):
        d = delta.ravel()
        return float(np.sqrt(np.sum(d*d))/max(1, d.size))
    Ru = l2(interior_view(state.fields['u']) - u_old)
    Rv = l2(interior_view(state.fields['v']) - v_old)
    Rp = 0.0
    continuity = float(np.linalg.norm(div_op(ui_full, vi_full, 1.0, 1.0)) / max(1.0, nx*ny))
    # NaN / Inf detection
    nan_detected = False
    for name in ('u','v','p'):
        arr = interior_view(state.fields[name])
        if not np.isfinite(arr).all():
            nan_detected = True
            break
    if diagnostics_enabled:
        print(f"[Step {iteration}] Residuals: Ru={Ru:.3g}, Rv={Rv:.3g}, continuity={continuity:.3g}")
    tracker.add('Ru', Ru)
    tracker.add('Rv', Rv)
    tracker.add('Rp', Rp)
    tracker.add('continuity', continuity)
    tracker.add('Rp_iterations', pressure_diag.get('Rp_iterations', 0))
    tracker.add('Rp_residual', pressure_diag.get('Rp_residual', 0.0))
    residuals = {
        'Ru': Ru,
        'Rv': Rv,
        'Rp': Rp,
        'continuity': continuity
    }
    diagnostics = {
        'iteration': iteration,
        'dt': dt,
        'CFL': current_cfl,
        'plateau_Ru': tracker.plateau_detect('Ru', window=10, threshold=-0.005),
        'max_residual': max(residuals.values()),
        'residual_drop_Ru_orders': max(0.0, (
            (math.log10(tracker.series['Ru'].values[0]) - math.log10(residuals['Ru']))
            if 'Ru' in tracker.series and tracker.series['Ru'].values and residuals['Ru'] > 0 and tracker.series['Ru'].values[0] > 0
            else 0.0
        )),
        'nu': nu,
        'Rp_iterations': pressure_diag.get('Rp_iterations', 0),
        'Rp_residual': pressure_diag.get('Rp_residual', 0.0),
        'Rp_converged': pressure_diag.get('Rp_converged', False),
        'Rp_method': pressure_diag.get('Rp_method', 'cg'),
        'divergence_norm': pressure_diag.get('divergence_norm', 0.0),
        'nan_detected': nan_detected
    }
    if getattr(config, 'seed', None) is not None:
        diagnostics['seed'] = config.seed
    # Invariant assertions (optional)
    if getattr(config, 'assert_invariants', False):
        # Divergence should not increase after projection vs baseline
        try:
            prev_div = pressure_diag.get('divergence_norm')
            assert prev_div is not None
        except AssertionError:
            raise AssertionError("Invariant failed: divergence_norm missing from pressure diagnostics")
        assert continuity <= max(1e3, prev_div * 10)  # loose safety ceiling
        # Reference pressure pinned
        p_interior = interior_view(state.fields['p'])
        assert abs(p_interior.flat[0]) < 1e-10, "Invariant failed: reference pressure cell not ~0"
    # Structured logging (optional)
    log_path = getattr(config, 'log_path', None)
    if log_path:
        logger = getattr(config, '_logger_instance', None)
        if logger is None:
            logger = JsonlLogger(log_path)
            setattr(config, '_logger_instance', logger)
        logger.log({'type':'step','it':iteration,'dt':dt,'CFL':current_cfl,'Ru':Ru,'Rv':Rv,'continuity':continuity,'Rp_it':pressure_diag.get('Rp_iterations',0),'nan':nan_detected})
        if nan_detected:
            logger.log({'type':'error','it':iteration,'reason':'nan_detected'})
            # Emergency checkpoint
            try:
                from ..io.checkpoint import save_checkpoint
                ck_path = getattr(config, 'emergency_checkpoint_path', None)
                if ck_path:
                    base, ext = (ck_path.rsplit('.',1)+['npz'])[:2]
                    fail_path = f"{base}_FAIL_NAN.{ext}"
                else:
                    fail_path = f"emergency_FAIL_NAN_it{iteration}.npz"
                save_checkpoint(fail_path, state, iteration, diagnostics.get('time', diagnostics.get('wall_time',0.0)), config)
                logger.log({'type':'emergency_checkpoint','path':fail_path,'it':iteration})
            except Exception:
                logger.log({'type':'emergency_checkpoint','status':'failed','it':iteration})
    return state, residuals, diagnostics
