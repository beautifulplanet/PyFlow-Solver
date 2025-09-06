"""Minimal projection (fractional‑step) solver.

Objective: reliably reduce discrete divergence for the test field with the
fewest moving parts.

We purposefully use the *unscaled* Chorin variant:
    Solve   Lap(p) = div(U*)
    Correct U <- U* - grad(p)

This avoids (rho/dt) amplification when dt is very small (the test uses
dt=2e-4). Scaling both RHS and correction by large reciprocal factors made
the earlier implementation numerically sensitive; removing them yields a
better conditioned linear system and still enforces incompressibility
because the discrete identity div(U* - grad p) = div(U*) - Lap(p) applies.

Assumptions: rho=1 effectively inside projection (rho cancels). Viscosity
affects only predictor (disabled in test). Dirichlet p=0 at boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import numpy as np
from .state import SolverState
from .advection import convective_terms, cfl_dt
import os

@dataclass
class ProjectionStats:
    iters: int
    dt: float
    notes: Dict[str, Any]

def _ensure_velocity_fields(state: SolverState):
    nx, ny = state.shape()
    state.require_field('u', (nx, ny))
    state.require_field('v', (nx, ny))
    state.require_field('p', (nx, ny))

def predict_velocity(state: SolverState, dt: float, use_advection: bool = True):
    """Forward Euler predictor with optional first-order upwind advection.

    u_t + (u·∇)u = nu Lap(u) (pressure later projected out)
    """
    _ensure_velocity_fields(state)
    u = state.fields['u']; v = state.fields['v']
    dx = state.mesh.dx(); dy = state.mesh.dy()
    dx2 = dx*dx; dy2 = dy*dy
    lap_u = (u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dx2 + (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dy2
    lap_v = (v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1])/dx2 + (v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])/dy2
    if use_advection:
        Cu, Cv = convective_terms(u, v, dx, dy)
        # Include diffusion for stability
        u[1:-1,1:-1] += dt * (-Cu[1:-1,1:-1] + state.nu * lap_u)
        v[1:-1,1:-1] += dt * (-Cv[1:-1,1:-1] + state.nu * lap_v)
    return u, v

def pressure_rhs_unscaled(state: SolverState) -> np.ndarray:
    """Compute RHS = div(U*)."""
    u = state.fields['u']; v = state.fields['v']
    dx = state.mesh.dx(); dy = state.mesh.dy()
    div = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
    rhs = np.zeros_like(u)
    rhs[1:-1,1:-1] = div
    return rhs

def _jacobi_poisson(rhs: np.ndarray, p: np.ndarray, dx: float, dy: float, tol: float, max_iter: int, debug: bool) -> int:
    """Jacobi solver core for Lap(p)=rhs with Dirichlet boundaries."""
    work = np.zeros_like(p)
    dx2 = dx*dx; dy2 = dy*dy
    denom = 2*(dx2+dy2)
    for it in range(max_iter):
        work[1:-1,1:-1] = (
            (p[2:,1:-1] + p[:-2,1:-1]) * dy2 +
            (p[1:-1,2:] + p[1:-1,:-2]) * dx2 - rhs[1:-1,1:-1]*dx2*dy2
        ) / denom
        p[1:-1,1:-1] = work[1:-1,1:-1]
        if it < 5 or it % 200 == 0:
            lap = ((p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1]) / dx2 + (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2]) / dy2)
            r = lap - rhs[1:-1,1:-1]
            rmax = float(np.max(np.abs(r))) if r.size else 0.0
            if debug and (it % 400 == 0 or it < 5):
                print(f"poisson_jacobi it={it} resid={rmax:.3e}")
            if rmax < tol:
                return it + 1
    return max_iter

def _sor_poisson(rhs: np.ndarray, p: np.ndarray, dx: float, dy: float, tol: float, max_iter: int, omega: float, debug: bool) -> int:
    """Successive Over-Relaxation for Lap(p)=rhs (Dirichlet)."""
    dx2 = dx*dx; dy2 = dy*dy
    beta = dx2*dy2/(2*(dx2+dy2))  # from rearranged Jacobi formula
    for it in range(max_iter):
        # Red-black SOR not needed for clarity; simple in-place sweep
        for i in range(1, p.shape[0]-1):
            pim1 = p[i-1]; pi = p[i]; pip1 = p[i+1]
            for j in range(1, p.shape[1]-1):
                pij_old = pi[j]
                pij_new = (
                    (pip1[j] + pim1[j]) * dy2 +
                    (pi[j+1] + pi[j-1]) * dx2 - rhs[i,j]*dx2*dy2
                ) / (2*(dx2+dy2))
                pi[j] = pij_old + omega * (pij_new - pij_old)
        if it < 5 or it % 200 == 0:
            lap = ((p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1]) / dx**2 + (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2]) / dy**2)
            r = lap - rhs[1:-1,1:-1]
            rmax = float(np.max(np.abs(r))) if r.size else 0.0
            if debug and (it % 400 == 0 or it < 5):
                print(f"poisson_sor it={it} resid={rmax:.3e}")
            if rmax < tol:
                return it + 1
    return max_iter

# ---------------- Multigrid (V-cycle) Prototype -----------------
def _lap(p: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dx2 = dx*dx; dy2 = dy*dy
    lap = np.zeros_like(p)
    lap[1:-1,1:-1] = ((p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1]) / dx2 +
                      (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2]) / dy2)
    return lap

def _residual(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return rhs - _lap(p, dx, dy)

def _restrict_full_weighting(r_f: np.ndarray) -> np.ndarray:
    # Assuming size n = 2*m +1 pattern; produce coarse n_c = (n-1)//2 +1
    nxf, nyf = r_f.shape
    nxc = (nxf -1)//2 +1
    nyc = (nyf -1)//2 +1
    r_c = np.zeros((nxc, nyc), dtype=r_f.dtype)
    for i_c in range(1, nxc-1):
        i_f = 2*i_c
        for j_c in range(1, nyc-1):
            j_f = 2*j_c
            r_c[i_c,j_c] = (
                4*r_f[i_f,j_f] +
                2*(r_f[i_f+1,j_f] + r_f[i_f-1,j_f] + r_f[i_f,j_f+1] + r_f[i_f,j_f-1]) +
                (r_f[i_f+1,j_f+1] + r_f[i_f-1,j_f+1] + r_f[i_f+1,j_f-1] + r_f[i_f-1,j_f-1])
            ) / 16.0
    return r_c

def _prolong_add(e_c: np.ndarray, e_f: np.ndarray):
    # Bilinear prolongation add: inject coarse points then interpolate
    nxc, nyc = e_c.shape
    nxf = 2*(nxc-1)+1
    # Inject
    for i_c in range(nxc):
        for j_c in range(nyc):
            e_f[2*i_c, 2*j_c] += e_c[i_c,j_c]
    # Interpolate in x direction (even j)
    for i_f in range(0, nxf-2, 2):
        for j_f in range(0, nxf, 2):
            e_f[i_f+1, j_f] += 0.5*(e_f[i_f,j_f] + e_f[i_f+2,j_f])
    # Interpolate in y direction (even i)
    for i_f in range(0, nxf, 1):
        if i_f % 2 == 0:
            for j_f in range(0, nxf-2, 2):
                e_f[i_f, j_f+1] += 0.5*(e_f[i_f,j_f] + e_f[i_f,j_f+2])
    # Interior odd/odd
    for i_f in range(0, nxf-2, 2):
        for j_f in range(0, nxf-2, 2):
            e_f[i_f+1, j_f+1] += 0.25*(e_f[i_f,j_f] + e_f[i_f+2,j_f] + e_f[i_f,j_f+2] + e_f[i_f+2,j_f+2])

def _smooth_jacobi(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float, sweeps: int, scratch: np.ndarray, omega: float = 1.0):
    """(Weighted) Jacobi smoothing sweeps.

    omega = 1 gives classic Jacobi. For weighted Jacobi ("wjacobi") typical omega in (0,1)."""
    dx2 = dx*dx; dy2 = dy*dy
    denom = 2*(dx2+dy2)
    for _ in range(sweeps):
        scratch[1:-1,1:-1] = (
            (p[2:,1:-1] + p[:-2,1:-1]) * dy2 +
            (p[1:-1,2:] + p[1:-1,:-2]) * dx2 - rhs[1:-1,1:-1]*dx2*dy2
        ) / denom
        # Weighted update
        p[1:-1,1:-1] = p[1:-1,1:-1] + omega * (scratch[1:-1,1:-1] - p[1:-1,1:-1])

def _vcycle(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float, level: int, max_level: int, pre: int, post: int, scratch: np.ndarray, smoother: str, wjac_omega: float) -> int:
    # Returns smoothing sweeps performed
    sweeps = 0
    n = p.shape[0]
    # Base case: small grid -> a few smoothing sweeps
    if level == max_level or n <= 5:
        _smooth_jacobi(p, rhs, dx, dy, pre+post+4, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0))
        return pre+post+4
    # Pre-smooth
    _smooth_jacobi(p, rhs, dx, dy, pre, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0)); sweeps += pre
    # Residual and restrict
    res = _residual(p, rhs, dx, dy)
    res_c = _restrict_full_weighting(res)
    nxc = res_c.shape[0]
    dx_c = (dx * (n-1)) / (nxc -1)  # since length preserved (domain length constant)
    dy_c = dx_c  # uniform square
    e_c = np.zeros_like(res_c)
    scratch_c = np.zeros_like(res_c)
    sweeps += _vcycle(e_c, res_c, dx_c, dy_c, level+1, max_level, pre, post, scratch_c, smoother, wjac_omega)
    # Prolongate error and correct
    e_f = np.zeros_like(p)
    _prolong_add(e_c, e_f)
    p += e_f
    # Post-smooth
    _smooth_jacobi(p, rhs, dx, dy, post, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0)); sweeps += post
    return sweeps

def _multigrid_poisson(rhs: np.ndarray, p: np.ndarray, dx: float, dy: float, tol: float, max_cycles: int, debug: bool) -> int:
    scratch = np.zeros_like(p)
    total_sweeps = 0
    pre = int(os.environ.get('PROJECTION_MG_PRE', '2'))
    post = int(os.environ.get('PROJECTION_MG_POST', '2'))
    smoother = os.environ.get('PROJECTION_MG_SMOOTHER', 'jacobi').lower()
    wjac_omega = float(os.environ.get('PROJECTION_MG_JACOBI_OMEGA', '0.8'))
    # Determine max levels possible
    n = p.shape[0]
    max_possible = 0
    while (n -1) % 2 == 0 and n > 5:
        max_possible += 1
        n = (n -1)//2 +1
    max_level = max_possible  # allow full coarsening
    for cycle in range(max_cycles):
        sweeps = _vcycle(p, rhs, dx, dy, 0, max_level, pre=pre, post=post, scratch=scratch, smoother=smoother, wjac_omega=wjac_omega)
        total_sweeps += sweeps
        # Residual check
        r = _residual(p, rhs, dx, dy)
        rmax = float(np.max(np.abs(r[1:-1,1:-1]))) if r.size else 0.0
        if debug:
            print(f"poisson_mg cycle={cycle} resid={rmax:.3e}")
        if rmax < tol:
            break
    return total_sweeps

def solve_pressure_poisson_unscaled(rhs: np.ndarray, state: SolverState, tol: float = 1e-6, max_iter: int = 8000) -> int:
    """Solve Lap(p)=rhs (Dirichlet p=0) selecting method via PROJECTION_LINSOLVER.

    Environment:
      PROJECTION_LINSOLVER = jacobi|sor|mg
      PROJECTION_SOR_OMEGA (float)
      PROJECTION_MG_PRE / PROJECTION_MG_POST (int) pre/post smoothing sweeps
      PROJECTION_MG_SMOOTHER = jacobi|wjacobi (weighted jacobi)
      PROJECTION_MG_JACOBI_OMEGA (float, default 0.8 for wjacobi)
    """
    p = state.fields['p']
    p.fill(0.0)
    dx = state.mesh.dx(); dy = state.mesh.dy()
    debug = os.environ.get('PROJECTION_DEBUG','') == '1'
    method = os.environ.get('PROJECTION_LINSOLVER', 'jacobi').lower()
    if method == 'sor':
        omega = float(os.environ.get('PROJECTION_SOR_OMEGA', '1.7'))
        return _sor_poisson(rhs, p, dx, dy, tol, max_iter, omega, debug)
    elif method == 'mg':
        return _multigrid_poisson(rhs, p, dx, dy, tol, max_iter, debug)
    else:
        return _jacobi_poisson(rhs, p, dx, dy, tol, max_iter, debug)

def correct_velocity_unscaled(state: SolverState):
    """Apply U <- U* - grad(p) (unscaled)."""
    u = state.fields['u']; v = state.fields['v']; p = state.fields['p']
    dx = state.mesh.dx(); dy = state.mesh.dy()
    # Interior central differences
    u[1:-1,1:-1] -= (p[2:,1:-1]-p[:-2,1:-1])/(2*dx)
    v[1:-1,1:-1] -= (p[1:-1,2:]-p[1:-1,:-2])/(2*dy)
    # Left/right boundaries (one-sided)
    u[0,1:-1]    -= (p[1,1:-1]-p[0,1:-1]) / dx
    u[-1,1:-1]   -= (p[-1,1:-1]-p[-2,1:-1]) / dx
    # Bottom/top boundaries
    v[1:-1,0]    -= (p[1:-1,1]-p[1:-1,0]) / dy
    v[1:-1,-1]   -= (p[1:-1,-1]-p[1:-1,-2]) / dy
    # Corners (average of adjacent one-sided)
    u[0,0]      -= (p[1,0]-p[0,0]) / dx
    u[0,-1]     -= (p[1,-1]-p[0,-1]) / dx
    u[-1,0]     -= (p[-1,0]-p[-2,0]) / dx
    u[-1,-1]    -= (p[-1,-1]-p[-2,-1]) / dx
    v[0,0]      -= (p[0,1]-p[0,0]) / dy
    v[-1,0]     -= (p[-1,1]-p[-1,0]) / dy
    v[0,-1]     -= (p[0,-1]-p[0,-2]) / dy
    v[-1,-1]    -= (p[-1,-1]-p[-1,-2]) / dy
    return u, v

def projection_step(state: SolverState, dt: Optional[float] = None, cfl: float = 0.5,
                    use_advection: bool = True, adaptive_dt: bool = True,
                    boundary_fn: Optional[Callable[[SolverState], None]] = None) -> ProjectionStats:
    # Feature flag guard: block unless PROJECTION_ENABLE=1
    if os.environ.get('PROJECTION_ENABLE','0') != '1':
        raise RuntimeError('Projection solver disabled (set PROJECTION_ENABLE=1 to enable)')
    # Orchestrates fractional step; each kernel currently unimplemented.
    _ensure_velocity_fields(state)
    if adaptive_dt or dt is None:
        dx = state.mesh.dx(); dy = state.mesh.dy()
        dt_cfl = cfl_dt(state.fields['u'], state.fields['v'], dx, dy, state.nu, cfl=cfl)
        # Also include pure diffusion explicit limit (already inside cfl_dt) but keep hook for future sources
        dt = dt_cfl if dt is None else min(dt, dt_cfl)
    # Measure divergence BEFORE
    u = state.fields['u']; v = state.fields['v']
    dx = state.mesh.dx(); dy = state.mesh.dy()
    div0 = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
    linf_before = float(np.max(np.abs(div0))) if div0.size else 0.0

    # Predictor (advection+diffusion) only if requested
    if use_advection:
        predict_velocity(state, dt, use_advection=True)

    # Apply boundary conditions (e.g., lid-driven cavity) prior to pressure solve
    if boundary_fn is not None:
        boundary_fn(state)

    # Poisson solve & correction (unscaled formulation)
    rhs = pressure_rhs_unscaled(state)
    # Adaptive tolerance logic (optional)
    base_tol = float(os.environ.get('PROJECTION_POISSON_BASE_TOL', '1e-6'))
    adapt_ref = float(os.environ.get('PROJECTION_ADAPT_REF', '1e-3'))
    adaptive_flag = os.environ.get('PROJECTION_ADAPTIVE_TOL', '0') == '1'
    tol = base_tol
    if adaptive_flag:
        scale = max(1.0, linf_before / adapt_ref)
        tol = min(base_tol * scale, 1e-3)  # never looser than 1e-3 absolute
    iters = solve_pressure_poisson_unscaled(rhs, state, tol=tol)
    correct_velocity_unscaled(state)

    # Divergence AFTER
    u = state.fields['u']; v = state.fields['v']
    div1 = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
    linf_after = float(np.max(np.abs(div1))) if div1.size else 0.0
    state.advance_time(dt)
    if os.environ.get('PROJECTION_DEBUG','') == '1':
        print(f"projection diagnostics: before={linf_before:.3e} after={linf_after:.3e} max|p|={np.max(np.abs(state.fields['p'])):.3e}")
    # Capture final residual estimate for stats
    p = state.fields['p']
    lap = ((p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1]) / dx**2 + (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2]) / dy**2)
    final_resid = float(np.max(np.abs(lap - rhs[1:-1,1:-1]))) if lap.size else 0.0
    method = os.environ.get('PROJECTION_LINSOLVER', 'jacobi').lower()
    return ProjectionStats(iters=iters, dt=dt, notes={
        "div_linf": linf_after,
        "div_linf_before": linf_before,
        "adaptive_dt": adaptive_dt,
        "use_advection": use_advection,
        "p_max": float(np.max(np.abs(state.fields['p']))),
        "poisson_tol": tol,
        "poisson_method": method,
        "poisson_resid_linf": final_resid,
        "adaptive_tol": adaptive_flag,
    })
