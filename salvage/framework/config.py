"""Configuration schema & validation.

Fail-fast design: missing required keys cause immediate ValidationError.
Extensible via optional sections; unknown keys recorded but not fatal (toggle strict later if desired).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

class ValidationError(Exception):
    pass

REQUIRED_KEYS = ["mesh.nx", "mesh.ny", "physics.case"]
OPTIONAL_DEFAULTS = {
    "mesh.lx": 1.0,
    "mesh.ly": 1.0,
    "physics.Re": 100.0,
    "time.dt": 1e-3,
    "time.max_steps": 1000,
    "numerics.scheme": "central",
    "numerics.cfl": 0.5,
}

@dataclass
class Config:
    data: Dict[str, Any]
    unknown: Dict[str, Any]

    def get(self, dotted: str) -> Any:
        return self.data[dotted]

    def maybe(self, dotted: str, default=None) -> Any:
        return self.data.get(dotted, default)


def flatten(inp: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inp.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def load_config(raw: Dict[str, Any], strict: bool = False) -> Config:
    flat = flatten(raw)
    # Apply defaults
    for k, v in OPTIONAL_DEFAULTS.items():
        flat.setdefault(k, v)
    # Validate required
    missing = [k for k in REQUIRED_KEYS if k not in flat]
    if missing:
        raise ValidationError(f"Missing required config keys: {missing}")
    # Partition unknown
    allowed = set(REQUIRED_KEYS) | set(OPTIONAL_DEFAULTS.keys())
    unknown = {k: v for k, v in flat.items() if k not in allowed}
    if strict and unknown:
        raise ValidationError(f"Unknown keys present under strict mode: {sorted(unknown)}")
    return Config(data=flat, unknown=unknown)
