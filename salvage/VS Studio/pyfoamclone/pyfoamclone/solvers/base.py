from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
from ..residuals.manager import ResidualManager


class GridLike(Protocol):  # minimal protocol for current placeholders
    nx: int
    ny: int


@dataclass
class SolverState:
    iteration: int = 0
    time: float = 0.0
    dt: float = 1e-2


class BaseSolver:
    def __init__(self, grid: GridLike, reynolds: float):
        if reynolds <= 0:
            raise ValueError("Reynolds must be positive")
        self.grid = grid
        self.reynolds = reynolds
        self.residuals = ResidualManager()
        self.state = SolverState()

    def initialize(self) -> None:  # pragma: no cover - abstract placeholder
        pass

    def step(self) -> None:  # pragma: no cover - abstract placeholder
        raise NotImplementedError

    def run(self, max_iter: int, tol: float) -> None:
        self.initialize()
        for _ in range(max_iter):
            self.step()
            self.state.iteration += 1
            # simple break on synthetic residual if present
            u_res = self.residuals.series.get("u")
            if u_res and u_res.last() < tol:
                break
