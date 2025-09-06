from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class SolverConfig:
    nx: int = 32
    ny: int = 32
    lx: float = 1.0
    ly: float = 1.0
    Re: float = 100.0
    solver: str = "pyfoam"  # or "finitude"
    max_iter: int = 1000
    tol: float = 1e-4
    cfl_target: float = 0.5
    cfl_growth: float = 1.1
    lin_tol: float = 1e-10
    lin_maxiter: int = 200
    lid_velocity: float = 1.0  # moving lid velocity for cavity benchmark
    enforce_projection_loops: int = 3  # extra pressure correction iterations
    disable_advection: bool = True  # stabilize tests by default
    test_mode: bool = True  # enables simplified projection for unit tests
    keep_lid_corners: bool = False  # allow optional full-lid top row

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolverConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

    @classmethod
    def load(cls, path: str | Path) -> "SolverConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
