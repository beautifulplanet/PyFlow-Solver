"""Simulation Driver abstraction.

Provides a reusable, front-end agnostic time‑stepping interface over the
validated solver core. Primary entry point: `SimulationDriver.run()`
which is a generator yielding a structured step payload containing the
state reference, residuals, diagnostics, and derived flow metrics.

Design Goals:
- Decouple simulation loop from UI (CLI / Dashboard / AI control).
- Provide consistent telemetry per step (residuals, divergence stats, timing).
- Support steady-state detection hooks while allowing external override.
- Keep state in-place (no deep copies) for performance; consumers must treat
  yielded state as read-only unless they intentionally modify the live state.

Extensibility Hooks:
- Pre/post step callbacks (e.g., for custom forcing, data taps, RL agents).
- Pluggable steady-state criterion object (future phase if needed).

Usage Example:
    from pyflow.driver.simulation_driver import SimulationDriver, DriverConfig
    driver = SimulationDriver(DriverConfig(max_steps=5000))
    for step_payload in driver.run():
        if step_payload['steady']:
            break

The driver does NOT perform visualization or CLI parsing; those are higher layers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import math
from typing import Callable, Dict, Any, Generator, Optional
import numpy as np

from ..core.ghost_fields import allocate_state, interior_view, State
from ..solvers.solver import step as core_step
from ..numerics.fluid_ops import divergence as div_op
from ..residuals.manager import ResidualManager

# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    Re: float = 100.0
    lid_velocity: float = 1.0
    disable_advection: bool = False
    advection_scheme: str = "quick"  # 'quick' | 'upwind'
    cfl_target: float = 0.5
    cfl_growth: float = 1.05
    lin_tol: float = 1e-10
    lin_maxiter: int = 800
    diagnostics: bool = False
    lx: float = 1.0
    ly: float = 1.0

@dataclass
class DriverConfig:
    nx: int = 65
    ny: int = 65
    max_steps: int = 10_000
    report_interval: int = 100
    steady_window: int = 100
    steady_slope_tol: float = 2e-3
    stop_on_steady: bool = True
    save_interval: int = 0  # placeholder for future snapshot system
    # Callback signatures: (driver, iteration, state) -> None / modifications in-place
    pre_step: Optional[Callable[["SimulationDriver", int, State], None]] = None
    post_step: Optional[Callable[["SimulationDriver", int, State], None]] = None

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def mean_free_stats(field: np.ndarray) -> tuple[float, float]:
    mean = float(field.mean())
    mf_norm = float(np.linalg.norm(field - mean))
    return mean, mf_norm

# ---------------------------------------------------------------------------
# Simulation Driver
# ---------------------------------------------------------------------------

class SimulationDriver:
    def __init__(self, driver_cfg: DriverConfig, physics_cfg: Optional[PhysicsConfig] = None):
        self.driver_cfg = driver_cfg
        self.physics_cfg = physics_cfg or PhysicsConfig()
        self.state: State = allocate_state(driver_cfg.nx, driver_cfg.ny)
        self.tracker = ResidualManager()
        self._start_wall: Optional[float] = None
        self._steady_detected: bool = False
        self._steady_iteration: Optional[int] = None

    # Public properties -----------------------------------------------------
    @property
    def steady_detected(self) -> bool:
        return self._steady_detected

    @property
    def steady_iteration(self) -> Optional[int]:
        return self._steady_iteration

    @property
    def elapsed(self) -> float:
        if self._start_wall is None:
            return 0.0
        return time.time() - self._start_wall

    # Core run generator ----------------------------------------------------
    def run(self) -> Generator[Dict[str, Any], None, None]:
        self._start_wall = time.time()
        cfg = self.driver_cfg
        physics = self.physics_cfg
        state = self.state
        tracker = self.tracker

        for it in range(cfg.max_steps):
            if cfg.pre_step:
                cfg.pre_step(self, it, state)

            # Advance one step using core solver
            state, residuals, diag = core_step(physics, state, tracker, it)

            # Divergence diagnostics
            ui = interior_view(state.fields['u'])
            vi = interior_view(state.fields['v'])
            div_full = div_op(ui, vi, state.meta['dx'], state.meta['dy'])
            div_mean, div_mf = mean_free_stats(div_full)

            steady = False
            steady_slope = None
            if len(tracker.series.get('Ru', []).values) >= cfg.steady_window:
                recent = np.array(tracker.series['Ru'].values[-cfg.steady_window:])
                if np.all(recent > 0):
                    xs = np.arange(recent.size)
                    logy = np.log10(recent)
                    steady_slope = float(np.polyfit(xs, logy, 1)[0])
                    if abs(steady_slope) < cfg.steady_slope_tol:
                        self._steady_detected = True
                        self._steady_iteration = it
                        steady = True
                        if cfg.stop_on_steady:
                            # Yield final step then break
                            payload = self._build_payload(it, residuals, diag, div_mean, div_mf, steady, steady_slope)
                            yield payload
                            break

            payload = self._build_payload(it, residuals, diag, div_mean, div_mf, steady, steady_slope)

            if cfg.post_step:
                cfg.post_step(self, it, state)

            # Periodic reporting
            if it % cfg.report_interval == 0 or (steady and cfg.stop_on_steady) or it == cfg.max_steps - 1:
                print(
                    f"Iter {it:5d} | dt={diag['dt']:.3e} CFL={diag['CFL']:.2f} "
                    f"Ru={residuals['Ru']:.3e} Rv={residuals['Rv']:.3e} div_mf={div_mf:.3e} div_mean={div_mean:.3e}" +
                    (f" steady|slope={steady_slope:.3e}" if steady else "")
                )
            yield payload

        # End loop (if not broken by steady)

    def _build_payload(self, iteration: int, residuals: Dict[str, float], diag: Dict[str, Any],
                       div_mean: float, div_mf: float, steady: bool, steady_slope: Optional[float]) -> Dict[str, Any]:
        return {
            'iteration': iteration,
            'state': self.state,  # reference (no copy)
            'residuals': residuals,
            'diagnostics': diag,
            'divergence_mean': div_mean,
            'divergence_mean_free_norm': div_mf,
            'steady': steady,
            'steady_slope': steady_slope,
            'elapsed': self.elapsed,
        }

# Convenience factory -------------------------------------------------------

def create_default_driver() -> SimulationDriver:
    return SimulationDriver(DriverConfig(), PhysicsConfig())

__all__ = [
    'PhysicsConfig',
    'DriverConfig',
    'SimulationDriver',
    'create_default_driver'
]
