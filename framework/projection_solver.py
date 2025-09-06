"""Projection solver (lightweight) with multigrid / Jacobi / SOR options.

Consolidated from salvage copy; exposes single public API surface used by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import numpy as np, os
from .state import SolverState
from .advection import convective_terms, cfl_dt

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
	_ensure_velocity_fields(state)
	u = state.fields['u']; v = state.fields['v']
	dx = state.mesh.dx(); dy = state.mesh.dy()
	dx2 = dx*dx; dy2 = dy*dy
	lap_u = (u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dx2 + (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dy2
	lap_v = (v[2:,1:-1]-2*v[1:-1,1:-1]+v[:-2,1:-1])/dx2 + (v[1:-1,2:]-2*v[1:-1,1:-1]+v[1:-1,:-2])/dy2
	if use_advection:
		Cu, Cv = convective_terms(u, v, dx, dy)
		u[1:-1,1:-1] += dt * (-Cu[1:-1,1:-1] + state.nu * lap_u)
		v[1:-1,1:-1] += dt * (-Cv[1:-1,1:-1] + state.nu * lap_v)
	return u, v

def pressure_rhs_unscaled(state: SolverState) -> np.ndarray:
	u = state.fields['u']; v = state.fields['v']
	dx = state.mesh.dx(); dy = state.mesh.dy()
	div = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
	rhs = np.zeros_like(u)
	rhs[1:-1,1:-1] = div
	return rhs

def _jacobi_poisson(rhs: np.ndarray, p: np.ndarray, dx: float, dy: float, tol: float, max_iter: int, debug: bool) -> int:
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
	dx2 = dx*dx; dy2 = dy*dy
	for it in range(max_iter):
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

def _lap(p: np.ndarray, dx: float, dy: float) -> np.ndarray:
	dx2 = dx*dx; dy2 = dy*dy
	lap = np.zeros_like(p)
	lap[1:-1,1:-1] = ((p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1]) / dx2 +
					  (p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2]) / dy2)
	return lap

def _residual(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
	return rhs - _lap(p, dx, dy)

def _restrict_full_weighting(r_f: np.ndarray) -> np.ndarray:
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
	nxc, nyc = e_c.shape
	nxf = 2*(nxc-1)+1
	for i_c in range(nxc):
		for j_c in range(nyc):
			e_f[2*i_c, 2*j_c] += e_c[i_c,j_c]
	for i_f in range(0, nxf-2, 2):
		for j_f in range(0, nxf, 2):
			e_f[i_f+1, j_f] += 0.5*(e_f[i_f,j_f] + e_f[i_f+2,j_f])
	for i_f in range(0, nxf, 1):
		if i_f % 2 == 0:
			for j_f in range(0, nxf-2, 2):
				e_f[i_f, j_f+1] += 0.5*(e_f[i_f,j_f] + e_f[i_f,j_f+2])
	for i_f in range(0, nxf-2, 2):
		for j_f in range(0, nxf-2, 2):
			e_f[i_f+1, j_f+1] += 0.25*(e_f[i_f,j_f] + e_f[i_f+2,j_f] + e_f[i_f,j_f+2] + e_f[i_f+2,j_f+2])

def _smooth_jacobi(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float, sweeps: int, scratch: np.ndarray, omega: float = 1.0):
	dx2 = dx*dx; dy2 = dy*dy
	denom = 2*(dx2+dy2)
	for _ in range(sweeps):
		scratch[1:-1,1:-1] = (
			(p[2:,1:-1] + p[:-2,1:-1]) * dy2 +
			(p[1:-1,2:] + p[1:-1,:-2]) * dx2 - rhs[1:-1,1:-1]*dx2*dy2
		) / denom
		p[1:-1,1:-1] = p[1:-1,1:-1] + omega * (scratch[1:-1,1:-1] - p[1:-1,1:-1])

def _vcycle(p: np.ndarray, rhs: np.ndarray, dx: float, dy: float, level: int, max_level: int, pre: int, post: int, scratch: np.ndarray, smoother: str, wjac_omega: float) -> int:
	n = p.shape[0]
	if level == max_level or n <= 5:
		_smooth_jacobi(p, rhs, dx, dy, pre+post+4, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0))
		return pre+post+4
	_smooth_jacobi(p, rhs, dx, dy, pre, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0))
	res = _residual(p, rhs, dx, dy)
	res_c = _restrict_full_weighting(res)
	nxc = res_c.shape[0]
	dx_c = (dx * (n-1)) / (nxc -1)
	dy_c = dx_c
	e_c = np.zeros_like(res_c)
	scratch_c = np.zeros_like(res_c)
	_vcycle(e_c, res_c, dx_c, dy_c, level+1, max_level, pre, post, scratch_c, smoother, wjac_omega)
	e_f = np.zeros_like(p)
	_prolong_add(e_c, e_f)
	p += e_f
	_smooth_jacobi(p, rhs, dx, dy, post, scratch, omega=(wjac_omega if smoother == 'wjacobi' else 1.0))
	return pre+post

def _multigrid_poisson(rhs: np.ndarray, p: np.ndarray, dx: float, dy: float, tol: float, max_cycles: int, debug: bool) -> int:
	scratch = np.zeros_like(p)
	pre = int(os.environ.get('PROJECTION_MG_PRE', '2'))
	post = int(os.environ.get('PROJECTION_MG_POST', '2'))
	smoother = os.environ.get('PROJECTION_MG_SMOOTHER', 'jacobi').lower()
	wjac_omega = float(os.environ.get('PROJECTION_MG_JACOBI_OMEGA', '0.8'))
	n = p.shape[0]
	max_possible = 0
	while (n -1) % 2 == 0 and n > 5:
		max_possible += 1
		n = (n -1)//2 +1
	max_level = max_possible
	for cycle in range(max_cycles):
		_vcycle(p, rhs, dx, dy, 0, max_level, pre=pre, post=post, scratch=scratch, smoother=smoother, wjac_omega=wjac_omega)
		r = _residual(p, rhs, dx, dy)
		rmax = float(np.max(np.abs(r[1:-1,1:-1]))) if r.size else 0.0
		if rmax < tol:
			return cycle+1
	return max_cycles

def solve_pressure_poisson_unscaled(rhs: np.ndarray, state: SolverState, tol: float = 1e-6, max_iter: int = 8000) -> int:
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
	u = state.fields['u']; v = state.fields['v']; p = state.fields['p']
	dx = state.mesh.dx(); dy = state.mesh.dy()
	u[1:-1,1:-1] -= (p[2:,1:-1]-p[:-2,1:-1])/(2*dx)
	v[1:-1,1:-1] -= (p[1:-1,2:]-p[1:-1,:-2])/(2*dy)
	u[0,1:-1]    -= (p[1,1:-1]-p[0,1:-1]) / dx
	u[-1,1:-1]   -= (p[-1,1:-1]-p[-2,1:-1]) / dx
	v[1:-1,0]    -= (p[1:-1,1]-p[1:-1,0]) / dy
	v[1:-1,-1]   -= (p[1:-1,-1]-p[1:-1,-2]) / dy
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
	if os.environ.get('PROJECTION_ENABLE','0') != '1':
		raise RuntimeError('Projection solver disabled (set PROJECTION_ENABLE=1 to enable)')
	_ensure_velocity_fields(state)
	if adaptive_dt or dt is None:
		dx = state.mesh.dx(); dy = state.mesh.dy()
		dt_cfl = cfl_dt(state.fields['u'], state.fields['v'], dx, dy, state.nu, cfl=cfl)
		dt = dt_cfl if dt is None else min(dt, dt_cfl)
	# Numerical safety clamps
	if dt is None or not np.isfinite(dt):  # pragma: no cover (defensive)
		dt = 1e-6
	dt = max(dt, 1e-12)
	u = state.fields['u']; v = state.fields['v']
	dx = state.mesh.dx(); dy = state.mesh.dy()
	div0 = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
	linf_before = float(np.max(np.abs(div0))) if div0.size else 0.0
	if use_advection:
		predict_velocity(state, dt, use_advection=True)
	if boundary_fn is not None:
		boundary_fn(state)
	# ------------------------------------------------------------------
	# Adaptive multi-pass projection refinement
	# ------------------------------------------------------------------
	max_passes = int(os.environ.get('PROJECTION_ADAPT_MAX_PASSES', '1'))
	if max_passes < 1:
		max_passes = 1
	target_abs = float(os.environ.get('PROJECTION_ADAPT_TARGET_ABS', '0'))  # 0 -> disabled
	rel_factor = float(os.environ.get('PROJECTION_ADAPT_TARGET_REL', '0'))  # 0 -> disabled
	accum_iters = 0
	linf_after = linf_before
	final_resid = 0.0
	method = os.environ.get('PROJECTION_LINSOLVER', 'jacobi').lower()
	passes_used = 0
	rhs = None
	# Initialize tol so it's always bound (updated each pass)
	base_tol_global = float(os.environ.get('PROJECTION_POISSON_BASE_TOL', '1e-6'))
	tol = base_tol_global
	for pidx in range(max_passes):
		passes_used = pidx + 1
		if boundary_fn is not None and pidx > 0:
			# Re-apply boundaries to prevent drift from successive corrections
			boundary_fn(state)
		rhs = pressure_rhs_unscaled(state)
		base_tol = base_tol_global
		adapt_ref = float(os.environ.get('PROJECTION_ADAPT_REF', '1e-3'))
		adaptive_flag = os.environ.get('PROJECTION_ADAPTIVE_TOL', '0') == '1'
		# Update loop-level tol
		tol = base_tol
		if adaptive_flag:
			scale = max(1.0, linf_after / adapt_ref)
			tol = min(base_tol * scale, 1e-3)
		iters = solve_pressure_poisson_unscaled(rhs, state, tol=tol)
		accum_iters += iters
		correct_velocity_unscaled(state)
		u = state.fields['u']; v = state.fields['v']
		div_pass = ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
		linf_after = float(np.max(np.abs(div_pass))) if div_pass.size else 0.0
		p_arr = state.fields['p']
		lap = ((p_arr[2:,1:-1]-2*p_arr[1:-1,1:-1]+p_arr[:-2,1:-1]) / dx**2 + (p_arr[1:-1,2:]-2*p_arr[1:-1,1:-1]+p_arr[1:-1,:-2]) / dy**2)
		final_resid = float(np.max(np.abs(lap - rhs[1:-1,1:-1]))) if lap.size else 0.0
		# Early exit criteria
		abs_ok = (target_abs > 0 and linf_after <= target_abs)
		rel_ok = (rel_factor > 0 and linf_before > 0 and linf_after <= linf_before * rel_factor)
		if abs_ok or rel_ok:
			break
	# Advance time after all refinement passes
	state.advance_time(dt)
	# Assertions (performed on final state)
	if os.environ.get('PROJECTION_ASSERT_REDUCTION','0') == '1' and linf_after >= linf_before and linf_before > 0:
		raise AssertionError(f"Projection failed to reduce divergence (before={linf_before} after={linf_after})")
	if os.environ.get('PROJECTION_ASSERT_RESID','0') == '1':
		factor = float(os.environ.get('PROJECTION_RESID_FACTOR','2.0'))
		# Use last tol from loop scope if defined else base
		if final_resid > factor * tol:
			raise AssertionError(f"Pressure residual {final_resid:.3e} exceeds allowed {factor}*tol ({factor*tol:.3e})")
	return ProjectionStats(iters=accum_iters, dt=dt, notes={
		"div_linf": linf_after,
		"div_linf_before": linf_before,
		"adaptive_dt": adaptive_dt,
		"use_advection": use_advection,
		"p_max": float(np.max(np.abs(state.fields['p']))),
		"poisson_method": method,
		"poisson_resid_linf": final_resid,
		"projection_passes": passes_used,
		"poisson_iters_total": accum_iters,
		"adapt_target_abs": target_abs,
		"adapt_target_rel": rel_factor,
	})
