from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import SolverConfig


def _manual_validate(data: Dict[str, Any]) -> None:
    required = ["schema_version", "nx", "ny", "lx", "ly", "Re", "solver", "max_iter", "tol"]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing key: {k}")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if data["nx"] <= 0 or data["ny"] <= 0:
        raise ValueError("nx/ny must be positive")
    if data["lx"] <= 0 or data["ly"] <= 0:
        raise ValueError("lx/ly must be > 0")
    if data["Re"] <= 0:
        raise ValueError("Re must be > 0")
    if data["solver"] not in {"pyfoam", "finitude", "physical"}:
        raise ValueError("solver invalid (expected pyfoam|finitude|physical)")
    # Optional linear solver controls
    for opt in ("lin_tol", "lin_maxiter"):
        if opt in data and data[opt] <= 0:
            raise ValueError(f"{opt} must be > 0")
    if data["max_iter"] <= 0:
        raise ValueError("max_iter must be > 0")
    if data["tol"] <= 0:
        raise ValueError("tol must be > 0")
    for k in ("relax_u", "relax_p"):
        if k in data and not (0 <= data[k] <= 1):
            raise ValueError(f"{k} outside [0,1]")


def load_config(path: str | Path) -> SolverConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    schema_path = Path(__file__).with_name("config_schema.json")
    # Legacy mapping: if a legacy solver label exists, translate before validation
    if data.get('solver') == 'synthetic_step':
        data['solver'] = 'physical'
    try:  # jsonschema optional
        import jsonschema  # type: ignore

        schema = json.loads(schema_path.read_text())
        try:
            jsonschema.validate(data, schema)  # type: ignore[arg-type]
        except jsonschema.ValidationError as e:  # map to ValueError for tests expecting ValueError
            raise ValueError(str(e))
    except ModuleNotFoundError:  # pragma: no cover - fallback path
        _manual_validate(data)
    return SolverConfig.from_dict(data)
