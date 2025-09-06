from .field import Field
import sys as _sys, os as _os
# Ensure repository root (containing 'framework') is on sys.path for direct package import.
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
if _repo_root not in _sys.path:
	_sys.path.insert(0, _repo_root)

# First import stubs for common types; we'll selectively override algorithms below.
from .stubs import (
	ValidationError,
	load_config,
	diffusion_residual,
	guard_synthetic,
	SyntheticUsageError,
	init_state,
	solve_steady_diffusion,
	Mesh,
	SolverState as _StubSolverState,
	projection_step as _proj_stub,
	cfl_dt as _cfl_stub,
	pressure_rhs_unscaled as _prhs_stub,
	solve_pressure_poisson_unscaled as _solve_p_stub,
)

# Default algorithm symbols point to stub versions (will be overwritten if framework available)
projection_step = _proj_stub
cfl_dt = _cfl_stub
pressure_rhs_unscaled = _prhs_stub
solve_pressure_poisson_unscaled = _solve_p_stub
SolverState = _StubSolverState

# Attempt to import real implementations; if successful overwrite names.
try:  # pragma: no cover
	from framework.state import SolverState as _RealSolverState, Mesh as _RealMesh
	from framework.projection_solver import (
		projection_step as _projection_step_real,
		pressure_rhs_unscaled as _pressure_rhs_unscaled_real,
		solve_pressure_poisson_unscaled as _solve_pressure_poisson_unscaled_real,
	)
	from framework.advection import cfl_dt as _cfl_dt_real
	SolverState = _RealSolverState  # type: ignore
	Mesh = _RealMesh  # type: ignore
	projection_step = _projection_step_real  # type: ignore
	pressure_rhs_unscaled = _pressure_rhs_unscaled_real  # type: ignore
	solve_pressure_poisson_unscaled = _solve_pressure_poisson_unscaled_real  # type: ignore
	cfl_dt = _cfl_dt_real  # type: ignore
except Exception:  # pragma: no cover
	pass

# Attempt to import real steady diffusion solver (used by convergence tests)
try:  # pragma: no cover
	from framework.steady_diffusion import solve_steady_diffusion as _solve_steady_diffusion_real
	solve_steady_diffusion = _solve_steady_diffusion_real  # type: ignore
except Exception:
	pass

# Expose which backend implementation we ended up binding (for tests / diagnostics)
PROJECTION_BACKEND = projection_step.__module__


__all__ = [
	'Field', 'ValidationError', 'load_config', 'diffusion_residual', 'guard_synthetic',
	'SyntheticUsageError', 'init_state', 'solve_steady_diffusion', 'projection_step',
	'cfl_dt', 'pressure_rhs_unscaled', 'solve_pressure_poisson_unscaled', 'Mesh', 'SolverState'
]
