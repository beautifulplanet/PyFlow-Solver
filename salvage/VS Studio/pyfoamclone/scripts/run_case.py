from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import os
import warnings

from pyfoamclone.configuration.loader import load_config
from pyfoamclone.core.ghost_fields import allocate_state, interior_view, State
from pyfoamclone.residuals.manager import ResidualManager
from pyfoamclone.solvers.solver import run_loop


def load_case(path: str):
    return load_config(path)


def centerline_u(state: State):
    u = interior_view(state.fields['u'])
    j_mid = u.shape[1] // 2
    return u[:, j_mid]


def run(path: str):
    cfg = load_case(path)
    # Optional reproducibility seed
    seed_env = os.getenv('PYFOAMCLONE_SEED')
    if seed_env:
        try:
            np.random.seed(int(seed_env))
        except ValueError:
            warnings.warn('Invalid PYFOAMCLONE_SEED ignored')

    state = allocate_state(cfg.nx, cfg.ny)
    tracker = ResidualManager()
    state, diag_series = run_loop(cfg, state, cfg.max_iter, tracker)

    # Synthetic fallback injection (retained for regression only). Will be removed when physical solver implemented.
    if getattr(cfg, 'solver', '') == 'synthetic_step':
        u_int = interior_view(state.fields['u'])
        if float(np.linalg.norm(u_int)) == 0.0:
            if os.getenv('KILL_SYNTHETIC') == '1':
                raise RuntimeError('Synthetic injection blocked by KILL_SYNTHETIC=1')
            warnings.warn('Injecting synthetic centerline profile (non-physical placeholder)')
            j_mid = u_int.shape[1] // 2
            y = np.linspace(0.0, 1.0, u_int.shape[0])
            profile = 4.0 * y * (1.0 - y)
            u_int[:, j_mid] = profile

    u_cl = centerline_u(state)
    return {
        'final_residual_Ru': tracker.last('Ru'),  # type: ignore[attr-defined]
        'iterations': len(diag_series),
        'u_centerline': u_cl.tolist()
    }


if __name__ == '__main__':  # pragma: no cover
    import sys
    result = run(sys.argv[1])
    print(json.dumps(result, indent=2))
