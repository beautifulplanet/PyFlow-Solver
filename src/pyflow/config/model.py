from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(ValueError):
    pass

AdvectionScheme = Literal['upwind','quick']

class SimulationConfig(BaseModel):
    # Core grid / physics
    nx: int = Field(..., ge=3, description="Number of interior u-cells in x (including boundary indices)")
    ny: int = Field(..., ge=3, description="Number of interior v-cells in y")
    Re: float = Field(100.0, gt=0.0, description="Reynolds number (non-dimensional)")
    lid_velocity: float = Field(1.0, description="Top lid tangential velocity")

    # Time stepping / CFL control
    cfl_target: float = Field(0.5, gt=0.0, le=1.0)
    cfl_growth: float = Field(1.05, gt=1.0, le=1.2)

    # Linear solver
    lin_tol: float = Field(1e-10, ge=1e-14, le=1e-2)
    lin_maxiter: int = Field(200, ge=1, le=10000)

    # Advection
    advection_scheme: AdvectionScheme = 'upwind'
    disable_advection: bool = False

    # Diagnostics / logging
    diagnostics: bool = True
    log_path: str | None = None
    log_stream: Any | None = None  # for in-memory tests; not hashed

    # Checkpointing
    checkpoint_interval: int | None = Field(None, ge=1)
    emergency_checkpoint_path: str | None = None

    # Reproducibility
    seed: int | None = Field(None, ge=0, le=2**32 - 1)

    # Versioning
    schema_version: int = 1

    # Reserved future fields (placeholders)
    turbulence_model: str | None = Field(None, description="Future turbulence closure identifier")

    @field_validator('advection_scheme')
    @classmethod
    def _check_scheme(cls, v, info):
        return v

    @model_validator(mode='after')
    def _cross_field(self):
        issues: list[str] = []
        if self.disable_advection and self.advection_scheme not in ('upwind','quick'):
            issues.append("disable_advection set but advection_scheme invalid")
        if self.log_path and self.log_stream is not None:
            issues.append("log_path and log_stream are mutually exclusive")
        ar = max(self.nx,1)/max(self.ny,1)
        if ar > 10 or ar < 0.1:
            # aspect ratio extreme, soft warning stored
            object.__setattr__(self, '_soft_warning', f"Extreme aspect ratio nx/ny={ar:.2f}")
        if issues:
            raise ConfigError("; ".join(issues))
        return self

    def hash_payload(self) -> dict:
        # Exclude ephemeral / runtime only fields
        data = self.model_dump()
        for k in ('log_stream',):
            data.pop(k, None)
        return data

    @property
    def config_hash(self) -> str:
        payload = self.hash_payload()
        blob = json.dumps(payload, sort_keys=True, separators=(',',':')).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def brief(self) -> dict:
        return {
            'nx': self.nx,
            'ny': self.ny,
            'Re': self.Re,
            'scheme': (None if self.disable_advection else self.advection_scheme),
            'lin_tol': self.lin_tol,
            'lin_maxiter': self.lin_maxiter,
            'cfl_target': self.cfl_target,
            'cfl_growth': self.cfl_growth,
            'hash': self.config_hash
        }

__all__ = ["ConfigError", "SimulationConfig"]