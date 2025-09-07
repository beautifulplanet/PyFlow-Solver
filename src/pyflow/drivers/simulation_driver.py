from __future__ import annotations

"""Reusable simulation driver for PyFlow.

Design Goals:
    * Provide a thin orchestration layer that repeatedly calls the low-level
        step() function while exposing a generator interface for incremental
    consumption (CLI, dashboard, AI control layer).
    * Keep zero policy about termination - caller decides when to stop based on
    yielded diagnostics (iterations, wall time, residuals, etc.).
    * Remain side-effect free beyond mutating the provided State instance.

Usage:
    driver = SimulationDriver(cfg, state, tracker)
    for state, residuals, diagnostics in driver.run(max_steps=500):
        # live monitoring / adaptive control
        if diagnostics['iteration'] % 50 == 0:
            print(diagnostics)
        if diagnostics['continuity'] < 1e-6:
            break

Future Extensions (non-breaking):
  * Callbacks (on_step_begin/end)
  * Event hooks (dt_adjusted, safeguard_triggered, plateau_detected)
  * Snapshot / checkpoint writer
  * Adaptive stopping policies packaged as utilities
"""
import time
from collections.abc import Callable, Generator
from typing import Any

from ..core.ghost_fields import State, allocate_state, interior_view
from ..io.checkpoint import save_checkpoint
from ..residuals.manager import ResidualManager
from ..solvers.solver import step


class SimulationDriver:
    def __init__(self,
                 config: Any,
                 state: State | None = None,
                 tracker: ResidualManager | None = None,
                 *,
                 allocate: Callable[[int, int], State] = allocate_state):
        self.config = config
        self.state = state
        self.tracker = tracker or ResidualManager()
        self.allocate_fn = allocate
        # Basic provenance / timing metadata
        self._t0 = None
        self.iteration = 0

    def ensure_state(self, nx: int, ny: int) -> State:
        if self.state is None:
            self.state = self.allocate_fn(nx, ny)
        return self.state

    def run(self,
            max_steps: int | None = None,
            *,
            start_iteration: int = 0,
            progress: bool = False,
            checkpoint_path: str | None = None,
            checkpoint_interval: int | None = None) -> Generator[tuple[State, dict[str, float], dict[str, Any]], None, None]:
        """Run the simulation, yielding after every solver step.

        Parameters
        ----------
        max_steps : int | None
            Maximum number of steps to perform (None => unbounded until caller breaks).
        start_iteration : int
            Override iteration counter start (useful when resuming).
        progress : bool
            If True, print a compact progress line each step.
        """
        self.iteration = start_iteration
        self._t0 = time.time()
        while True:
            if max_steps is not None and self.iteration >= max_steps:
                return
            if self.state is None:
                nx = getattr(self.config, 'nx', None)
                ny = getattr(self.config, 'ny', None)
                if nx is None or ny is None:
                    raise ValueError("SimulationDriver requires either an initial State or config.nx & config.ny for allocation")
                self.state = self.allocate_fn(nx, ny)
            s = self.state
            state, residuals, diagnostics = step(self.config, s, self.tracker, self.iteration)
            try:
                ui = interior_view(state.fields['u'])
                nyi, nxi = ui.shape
                cx = nxi // 2
                u_center = ui[:, cx]
                ly_val = getattr(self.config, 'ly', float(nyi - 1))
                y_coords = [j * (ly_val / (nyi - 1)) for j in range(nyi)] if nyi > 1 else [0.0]
                diagnostics['u_centerline'] = [y_coords, u_center.tolist()]
            except Exception:  # pragma: no cover
                pass
            diagnostics['wall_time'] = time.time() - self._t0
            if progress:
                print(f"iter={self.iteration} dt={diagnostics.get('dt'):.3g} CFL={diagnostics.get('CFL'):.3g} continuity={residuals.get('continuity'):.3e}")
            yield state, residuals, diagnostics
            if checkpoint_path and checkpoint_interval and checkpoint_interval > 0:
                if (self.iteration % checkpoint_interval) == 0:
                    try:
                        save_checkpoint(checkpoint_path, state, self.iteration, diagnostics.get('time', diagnostics.get('wall_time', 0.0)), self.config)
                        diagnostics['checkpoint_written'] = True
                    except Exception:  # pragma: no cover
                        diagnostics['checkpoint_written'] = False
            self.iteration += 1

__all__ = ["SimulationDriver"]
