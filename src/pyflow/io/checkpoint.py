from __future__ import annotations

"""Lightweight simulation checkpoint I/O.

Provides periodic persistence of full simulation state (fields + metadata)
to enable postmortem analysis and restart after failure.

Format: NumPy .npz (portable, compressed) with keys:
  meta: JSONserializable metadata dict (iteration, time, config hash)
  field_<name>: ndarray including ghost layers

Design Goals:
  * Zero external dependencies beyond NumPy
  * Atomic write (write to temp then rename) to avoid partial corruption
  * Backwards compatible  absence of file or fields handled gracefully
"""
import hashlib
import io
import json
import os
import tempfile
from typing import Any, cast

import numpy as np

from ..core.ghost_fields import State

try:  # optional import (new config system)
    from ..config.model import SimulationConfig
except Exception:  # pragma: no cover
    SimulationConfig = None  # type: ignore


def _hash_config(cfg: Any) -> str:
    try:
        if hasattr(cfg, '__dict__'):
            data = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
        else:
            data = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]
    except Exception:
        return 'unknown'


def _hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha1(arr.tobytes()).hexdigest()
    return h[:16]


def save_checkpoint(path: str, state: State, iteration: int, sim_time: float, cfg: Any) -> str:
    """Atomically persist simulation state.

    We write to a temporary file, flush & fsync, then os.replace to final path.
    This avoids partially written (zerolength / truncated) archives that can
    surface as EOFError during np.load under fast successive read/write cycles.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    field_hashes = {k: _hash_array(v) for k, v in state.fields.items()}
    cfg_hash = None
    if 'SimulationConfig' in globals() and isinstance(cfg, SimulationConfig):  # type: ignore
        try:
            cfg_hash = cfg.config_hash  # type: ignore
        except Exception:
            cfg_hash = None
    meta: dict[str, Any] = {
        'iteration': iteration,
        'time': sim_time,
        'nx': state.nx,
        'ny': state.ny,
        'config_hash': _hash_config(cfg),
        'structured_config_hash': cfg_hash,
        'field_hashes': field_hashes,
        'seed': getattr(cfg, 'seed', None),
        'schema_version': 1,
    }
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path) or '.')
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            arrays = {f"field_{k}": v for k, v in state.fields.items()}
            saver = cast(Any, np.savez_compressed)
            saver(f, meta=json.dumps(meta), **arrays)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return path


def load_checkpoint(path: str) -> tuple[State, dict[str, Any]]:
    with np.load(path) as data:
        meta = json.loads(str(data['meta']))
        fields = {k.removeprefix('field_'): data[k] for k in data.files if k.startswith('field_')}
        st = State(nx=meta.get('nx'), ny=meta.get('ny'), fields={k: v.copy() for k, v in fields.items()})
    return st, meta

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_checkpoint_bytes",
    "load_checkpoint_bytes",
]

def save_checkpoint_bytes(state: State, iteration: int, sim_time: float, cfg: Any) -> bytes:
    field_hashes = {k: _hash_array(v) for k,v in state.fields.items()}
    meta = {"iteration": iteration, "time": sim_time, "nx": state.nx, "ny": state.ny, "config_hash": _hash_config(cfg), "field_hashes": field_hashes, "seed": getattr(cfg,"seed",None), "schema_version":1}
    bio = io.BytesIO()
    arrays = {f"field_{k}": v for k,v in state.fields.items()}
    saver = cast(Any, np.savez_compressed)
    saver(bio, meta=json.dumps(meta), **arrays)
    return bio.getvalue()

def load_checkpoint_bytes(buf: bytes) -> tuple[State, dict[str, Any]]:
    bio = io.BytesIO(buf)
    with np.load(bio) as data:
        meta = json.loads(str(data['meta']))
        fields = {k.removeprefix('field_'): data[k] for k in data.files if k.startswith('field_')}
        st = State(nx=meta.get('nx'), ny=meta.get('ny'), fields={k: v.copy() for k,v in fields.items()})
    return st, meta
