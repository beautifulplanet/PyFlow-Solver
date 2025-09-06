from __future__ import annotations

from typing import Callable, Dict, Any

BCFunc = Callable[[Any], None]

_registry: Dict[str, BCFunc] = {}


def register(name: str):
    def deco(func: BCFunc) -> BCFunc:
        _registry[name] = func
        return func
    return deco


def get(name: str) -> BCFunc:
    return _registry[name]


def available() -> Dict[str, BCFunc]:
    return dict(_registry)


# Example placeholders
@register("wall")
def wall_bc(field):  # pragma: no cover - trivial placeholder
    field[...] = 0.0


@register("moving_wall")
def moving_wall_bc(field):  # pragma: no cover - trivial placeholder
    field[...] = 1.0
