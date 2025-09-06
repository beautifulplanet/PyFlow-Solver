import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Mapping


@dataclass
class State:
    """Container for simulation fields and lightweight metadata.

    Provides dict-like access for a transition period while new code should
    prefer explicit .fields access. Metadata lives in .meta (e.g. dt_prev).
    """
    nx: int
    ny: int
    fields: Dict[str, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    def arr(self, name: str) -> np.ndarray:
        return self.fields[name]

    # Backward compatibility (state['u'])
    def __getitem__(self, key: str) -> np.ndarray:  # type: ignore[override]
        return self.fields[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:  # type: ignore[override]
        self.fields[key] = value

    def get(self, key: str, default=None):  # for legacy lookups
        if key in self.fields:
            return self.fields[key]
        return self.meta.get(key, default)

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def keys(self):
        return self.fields.keys()

    def items(self):
        return self.fields.items()

    def values(self):
        return self.fields.values()


def allocate_state(nx: int, ny: int, fields: Tuple[str, ...] = ("u", "v", "p")) -> State:
    """Allocate a State with named fields including ghost cells.

    Parameters
    ----------
    nx, ny : interior cell counts.
    fields : iterable of field names to allocate.
    """
    shape = (ny + 2, nx + 2)
    arrs = {k: np.zeros(shape, dtype=float) for k in fields}
    return State(nx=nx, ny=ny, fields=arrs)


def interior_view(arr: np.ndarray) -> np.ndarray:
    return arr[1:-1, 1:-1]


def ghost_shapes(state: Mapping[str, np.ndarray] | State) -> Dict[str, Tuple[int, int]]:
    if isinstance(state, State):
        items = state.fields.items()
    else:
        items = state.items()
    return {k: v.shape for k, v in items if isinstance(v, np.ndarray)}
