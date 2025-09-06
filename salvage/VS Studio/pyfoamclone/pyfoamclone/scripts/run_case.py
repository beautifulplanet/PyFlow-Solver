from __future__ import annotations

import json
from pathlib import Path
import os
import numpy as np

from ..configuration.loader import load_config
from ..core.ghost_fields import allocate_state, interior_view, State
from ..residuals.manager import ResidualManager
from ..solvers.solver import run_loop
from .. import build_run_manifest


class RunResult(dict):  # supports dict access + tuple unpacking
    def __init__(self, state, manifest, tracker, u_centerline, y_centerline):
        super().__init__()
        self.state = state
        self.manifest = manifest
        self["manifest"] = manifest
        self["final_residual_Ru"] = tracker.last('Ru')
        self["iterations"] = manifest.get('iterations')
        self["u_centerline"] = u_centerline
        self["y_centerline"] = y_centerline

    def __iter__(self):  # unpack -> state, manifest
        yield self.state
        yield self.manifest

    def __len__(self):  # for safety in unpacking
        return 2


def load_case(path: str):
    return load_config(path)


def centerline_u(state: State):
    u = interior_view(state.fields['u'])
    j_mid = u.shape[1] // 2
    return u[:, j_mid]

def centerline_y(state: State, ly: float):
    u = interior_view(state.fields['u'])
    ny = u.shape[0]
    # y=0 at bottom, y=ly at top
    # Ensure y=0 is bottom (row 0), y=ly is top (row ny-1)
    return np.linspace(0.0, ly, ny)[::-1]


def run(path: str):
    """Primary interface returning (state, manifest) as expected by tests.

    A legacy dict style helper is provided for backward compatibility.
    """
    cfg = load_case(path)
    state = allocate_state(cfg.nx, cfg.ny)
    tracker = ResidualManager()
    state, diag_series = run_loop(cfg, state, cfg.max_iter, tracker)
    if getattr(cfg, 'solver', '') == 'synthetic_step':
        raise RuntimeError('synthetic_step solver removed: use solver="physical" or "pyfoam"')
    manifest = build_run_manifest(cfg.to_dict(), extra={'iterations': len(diag_series)})
    u_cl = centerline_u(state).tolist()
    y_cl = centerline_y(state, cfg.ly).tolist()
    return RunResult(state, manifest, tracker, u_cl, y_cl)


def run_legacy(path: str):  # pragma: no cover - compatibility shim
    rr = run(path)
    # already dict-like with needed keys
    return rr


if __name__ == '__main__':  # pragma: no cover
    import sys
    result = run(sys.argv[1])
    print(json.dumps(result, indent=2))
