from __future__ import annotations

from .registry import get
from typing import Dict


def apply_all(bc_map: Dict[str, str], state: Dict[str, object]):  # pragma: no cover simple
    for field_name, bc_name in bc_map.items():
        func = get(bc_name)
        func(state[field_name])
