from __future__ import annotations

import numpy as np

def convective_terms(u: np.ndarray, v: np.ndarray, dx: float, dy: float):
	"""Compute (u·∇)u and (u·∇)v with simple first-order upwind/central blend.

	Returns Cu, Cv arrays same shape as inputs.
	"""
	Cu = np.zeros_like(u)
	Cv = np.zeros_like(v)
	# Interior central differences for simplicity
	dudx = (u[2:,1:-1] - u[:-2,1:-1]) / (2*dx)
	dudy = (u[1:-1,2:] - u[1:-1,:-2]) / (2*dy)
	dvdx = (v[2:,1:-1] - v[:-2,1:-1]) / (2*dx)
	dvdy = (v[1:-1,2:] - v[1:-1,:-2]) / (2*dy)
	ui = u[1:-1,1:-1]; vi = v[1:-1,1:-1]
	Cu[1:-1,1:-1] = ui * dudx + vi * dudy
	Cv[1:-1,1:-1] = ui * dvdx + vi * dvdy
	return Cu, Cv

def cfl_dt(u: np.ndarray, v: np.ndarray, dx: float, dy: float, nu: float, cfl: float = 0.5) -> float:
	umax = float(np.max(np.abs(u))) if u.size else 0.0
	vmax = float(np.max(np.abs(v))) if v.size else 0.0
	velmax = max(umax, vmax, 1e-12)
	dt_adv = cfl * min(dx, dy) / velmax
	# Diffusive explicit limit (2D) ~ 0.25 * h^2 / nu
	dt_diff = 0.25 * min(dx, dy)**2 / max(nu, 1e-12)
	return float(min(dt_adv, dt_diff))
__all__ = ["convective_terms", "cfl_dt"]

