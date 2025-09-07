from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..core.ghost_fields import State, interior_view
from ..linear_solvers.interface import solve
from ..linear_solvers.preconditioners import jacobi_preconditioner
from ..numerics.core_ops_coherent import divergence_coherent as divergence
from ..numerics.core_ops_coherent import gradient_coherent as gradient
from ..numerics.core_ops_coherent import laplacian_coherent


def assemble_negative_laplacian(nx: int, ny: int, dx: float, dy: float):
    """Assemble sparse matrix A = -Laplace operator (SPD) with homogeneous Neumann-like
    interior stencil (no special BC rows except reference pressure fix).
    Center coefficient: 2*(1/dx^2 + 1/dy^2); neighbors: -1/dx^2 or -1/dy^2.
    A maps flattened p (row-major with index k = j*nx + i).
    """
    N = nx * ny
    A = sp.lil_matrix((N, N))
    for j in range(ny):
        for i in range(nx):
            idx = j*nx + i
            if (i == 0 or i == nx-1 or j == 0 or j == ny-1):
                if idx == 0:
                    # Reference cell: Dirichlet (fix p=0)
                    A[idx, idx] = 1.0
                else:
                    # Pure Neumann: identity row (no pressure change at boundary)
                    A[idx, idx] = 1.0
            else:
                # Interior: standard 5-point Laplacian
                A[idx, idx] = 2.0 / dx**2 + 2.0 / dy**2
                A[idx, idx-1] = -1.0 / dx**2
                A[idx, idx+1] = -1.0 / dx**2
                A[idx, idx-nx] = -1.0 / dy**2
                A[idx, idx+nx] = -1.0 / dy**2
    return A.tocsr()

def solve_pressure_poisson(state: State, dt: float, dx: float, dy: float, cfg, preconditioner=None, rhs_override=None):
    """Perform pressure projection to reduce velocity divergence.

    Modes
    -----
    Normal projection (rhs_override is None):
        Solve A p = -(1/dt) div(u*) with A = -Laplace (Neumann + pinned reference cell).
        Enforce mean-zero RHS for compatibility; remove constant mode by pinning.
    Manufactured / analytic verification (rhs_override provided):
        Treat RHS as belonging to Dirichlet Poisson problem ∇² p = RHS with p=0 on boundary.
        Assemble positive Laplacian with identity rows on boundary cells.
    """
    nx = interior_view(state.fields['u']).shape[1]
    ny = interior_view(state.fields['u']).shape[0]
    manufactured_mode = rhs_override is not None
    # Unified verbosity flag (may be disabled by CLI json-stream via force_quiet)
    verbose = getattr(cfg, 'diagnostics', True) and not getattr(cfg, 'force_quiet', False)

    # MATRIX ASSEMBLY / LINEAR OPERATOR
    if manufactured_mode:
        # Manufactured / Dirichlet Laplacian solve
        N = nx * ny
        boundary_mask = np.zeros((ny, nx), dtype=bool)
        boundary_mask[0, :] = True; boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True; boundary_mask[:, -1] = True
        b_idx = np.flatnonzero(boundary_mask.reshape(-1))

        class _PosLaplacianDirichlet(spla.LinearOperator):  # pragma: no cover
            def __init__(self):
                super().__init__(dtype=np.float64, shape=(N, N))
            def _matvec(self, x):
                field = x.reshape(ny, nx)
                lap = laplacian_coherent(field, dx, dy).reshape(-1)
                out = lap
                out[b_idx] = x[b_idx]  # identity rows enforce p=0 (given RHS boundary zeros)
                return out

        A = _PosLaplacianDirichlet()
        rhs = np.copy(rhs_override).reshape(-1)
        lin_tol = getattr(cfg, 'lin_tol', 1e-10)
        lin_maxiter = getattr(cfg, 'lin_maxiter', 500)
        solve_kwargs = {}
        if preconditioner is not None:
            solve_kwargs['M'] = preconditioner
        # (Skip automatic Jacobi for manufactured mode to keep it simple.)
        res_p = solve(A, rhs, method='cg', tol=lin_tol, maxiter=lin_maxiter, **solve_kwargs)
        p = res_p.x.reshape(ny, nx)
        interior_view(state.fields['p'])[:] = p
        # Diagnostics (treat residual of linear system akin to divergence metric)
        div_norm = float(np.linalg.norm(A @ p.reshape(-1) - rhs))
        div_before_norm = None
        div_after_raw_norm = None
    else:
        # Normal projection mode (Neumann + pinned reference)
        N = nx * ny
        class _NegLaplacianPinned(spla.LinearOperator):  # pragma: no cover
            def __init__(self):
                super().__init__(dtype=np.float64, shape=(N, N))
            def _matvec(self, x):
                field = x.reshape(ny, nx)
                lap = laplacian_coherent(field, dx, dy)
                out = (-lap).reshape(-1)
                out[0] = x[0]  # pin reference cell (remove null space)
                return out

        A = _NegLaplacianPinned()
        ui = interior_view(state.fields['u'])
        vi = interior_view(state.fields['v'])

        if verbose:
            print("=== PRESSURE PROJECTION DIAGNOSTICS ===")
            print("u* (before projection):\n", ui)
            print("v* (before projection):\n", vi)

        div_u = divergence(ui, vi, dx, dy)
        if verbose:
            print("div(u*) before projection:\n", div_u)
        div_before_norm = float(np.linalg.norm(div_u))
        if verbose:
            print(f"DIVERGENCE_NORM_BEFORE={div_before_norm}")

        rhs = (-div_u / max(dt, 1e-14)).reshape(-1)
        if verbose:
            print("RHS for Poisson (should be -div(u*)/dt):\n", rhs.reshape(ui.shape))
        rhs_mean = float(np.mean(rhs))
        rhs -= rhs_mean
        rhs[0] = 0.0
        if verbose:
            print(f"[pressure_solver] RHS mean subtracted (compatibility) = {rhs_mean:.6e}")
            print(f"DEBUG: Norm of RHS (divergence) = {np.linalg.norm(rhs)}")
            print(f"DEBUG: Velocity norm BEFORE correction: {np.linalg.norm(ui)}, {np.linalg.norm(vi)}")

        lin_tol = getattr(cfg, 'lin_tol', 1e-10)
        lin_maxiter = getattr(cfg, 'lin_maxiter', 400)
        solve_kwargs = {}
        if preconditioner is not None:
            solve_kwargs['M'] = preconditioner
        elif getattr(cfg, 'enable_jacobi_pc', True):  # default ON for acceleration phase
            try:
                Amat = assemble_negative_laplacian(nx, ny, dx, dy)
                Amat = sp.csr_matrix(Amat)  # ensure spmatrix subtype
                solve_kwargs['M'] = jacobi_preconditioner(Amat)
            except Exception:  # pragma: no cover - preconditioner optional
                pass
        res_p = solve(A, rhs, method='cg', tol=lin_tol, maxiter=lin_maxiter, **solve_kwargs)
        p = res_p.x.reshape(ny, nx)
        p -= p.flat[0]
        interior_view(state.fields['p'])[:] = p
        if verbose:
            print("Pressure solution p:\n", p)

        p_field = interior_view(state.fields['p'])
        dpdx, dpdy = gradient(p_field, dx, dy)
        ui_before = ui.copy(); vi_before = vi.copy()
        ui[:] -= dt * dpdx
        vi[:] -= dt * dpdy
        if verbose:
            print("u after projection: min=", np.min(ui), "max=", np.max(ui))
            print("v after projection: min=", np.min(vi), "max=", np.max(vi))
            print("Change in u: min=", np.min(ui - ui_before), "max=", np.max(ui - ui_before))
            print("Change in v: min=", np.min(vi - vi_before), "max=", np.max(vi - vi_before))
            print(f"DEBUG: Velocity norm AFTER correction: {np.linalg.norm(ui)}, {np.linalg.norm(vi)}")

        div_after = divergence(ui, vi, dx, dy)
        div_after_raw_norm = float(np.linalg.norm(div_after))
        if verbose:
            print(f"DIVERGENCE_NORM_AFTER_RAW={div_after_raw_norm}")
        div_norm = float(np.linalg.norm(div_after))
        if verbose:
            print(f"DIVERGENCE_NORM_FINAL={div_norm}")

    # Linear residual diagnostics
    try:
        pvec = p.reshape(-1)
        Ap = A @ pvec
        lin_residual_vec = Ap - rhs
        lin_residual_norm = float(np.linalg.norm(lin_residual_vec))
        rhs_norm = float(np.linalg.norm(rhs)) or 1.0
        lin_residual_rel = lin_residual_norm / rhs_norm
        if verbose:
            print(f"LIN_RESIDUAL_NORM={lin_residual_norm}")
            print(f"LIN_RESIDUAL_REL={lin_residual_rel}")
    except Exception as e:  # pragma: no cover
        lin_residual_norm = getattr(res_p, 'residual_norm', 0.0)
        rhs_norm = float(np.linalg.norm(rhs)) or 1.0
        lin_residual_rel = lin_residual_norm / rhs_norm
        print("[pressure_solver] Residual computation fallback:", e)

    rp_iters = getattr(res_p, 'iterations', 0)
    lin_tol = getattr(cfg, 'lin_tol', 1e-10)
    rp_residual = getattr(res_p, 'residual_norm', 0.0)
    if rp_iters == 0 and rp_residual > lin_tol:
        rp_iters = 1

    red_factor = None
    try:
        _div_before = locals().get('div_before_norm')
        _div_final = locals().get('div_norm')
        if _div_before not in (None,) and _div_final not in (None, 0.0):
            red_factor = _div_before / _div_final
    except Exception:
        red_factor = None

    diagnostics = {
        'Rp_iterations': rp_iters,
        'Rp_residual': rp_residual,
        'Rp_converged': getattr(res_p, 'converged', False),
        'Rp_method': getattr(res_p, 'method', 'cg'),
        'divergence_norm': div_norm,
        'lin_residual_norm': lin_residual_norm,
        'lin_residual_rel': lin_residual_rel,
        'divergence_reduction_factor': red_factor
    }
    _locals = locals()
    diagnostics['divergence_norm_initial'] = _locals.get('div_before_norm')
    diagnostics['divergence_norm_after_raw'] = _locals.get('div_after_raw_norm')
    diagnostics['divergence_norm_final'] = _locals.get('div_norm')
    return p, diagnostics
"""Deduplicated: removed second copy of functions (original retained above)."""
