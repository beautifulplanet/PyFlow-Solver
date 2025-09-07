from __future__ import annotations

"""Schema definitions for AI control layer.

Defines a structured representation of a simulation request that can be
populated from natural language and then mapped to CLI arguments.
"""
from dataclasses import dataclass, field
from typing import Any

SUPPORTED_PROBLEMS = {"lid_cavity": "Lid-driven cavity flow"}

@dataclass
class SimulationRequest:
    problem: str = "lid_cavity"  # problem identifier
    nx: int = 64
    ny: int = 64
    Re: float = 100.0
    lid_velocity: float = 1.0
    steps: int = 200
    scheme: str = "quick"  # advection scheme
    cfl: float = 0.5
    cfl_growth: float = 1.05
    diagnostics: bool = False
    json_stream: bool = True
    continuity_threshold: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_cli_args(self) -> list[str]:
        args = [
            f"--nx={self.nx}", f"--ny={self.ny}", f"--re={self.Re}", f"--lid-velocity={self.lid_velocity}",
            f"--steps={self.steps}", f"--scheme={self.scheme}", f"--cfl={self.cfl}", f"--cfl-growth={self.cfl_growth}",
        ]
        if self.diagnostics:
            args.append("--diagnostics")
        if self.json_stream:
            args.append("--json-stream")
        if self.continuity_threshold is not None:
            args.append(f"--continuity-threshold={self.continuity_threshold}")
        # Extra passthrough
        for k, v in self.extra.items():
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            else:
                args.append(f"--{k}={v}")
        return args

__all__ = ["SUPPORTED_PROBLEMS", "SimulationRequest"]
