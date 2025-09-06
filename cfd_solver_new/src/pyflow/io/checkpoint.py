from __future__ import annotations
"""Lightweight simulation checkpoint I/O.

Provides periodic persistence of full simulation state (fields + metadata)
to enable post‑mortem analysis and restart after failure.

Format: NumPy .npz (portable, compressed) with keys:
  meta: JSON‑serializable metadata dict (iteration, time, config hash)
  field_<name>: ndarray including ghost layers

Design Goals:
  * Zero external dependencies beyond NumPy
  * Atomic write (write to temp then rename) to avoid partial corruption
  * Backwards compatible – absence of file or fields handled gracefully
"""
from dataclasses import asdict
import json
import os
import tempfile
import hashlib
from typing import Any, Dict, cast
import numpy as np

from ..core.ghost_fields import State

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
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    field_hashes = {k: _hash_array(v) for k, v in state.fields.items()}
    meta: Dict[str, Any] = {
        'iteration': iteration,
        'time': sim_time,
        'nx': state.nx,
        'ny': state.ny,
        'config_hash': _hash_config(cfg),
        'field_hashes': field_hashes,
        'seed': getattr(cfg, 'seed', None),
        'schema_version': 1,
    }
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(path) or '.')
    os.close(tmp_fd)
    try:
        arrays = {f"field_{k}": v for k, v in state.fields.items()}
        # Use a cast to silence strict type analysis confusion about **kwargs
        saver = cast(Any, np.savez_compressed)
        saver(tmp_path, meta=json.dumps(meta), **arrays)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return path

def load_checkpoint(path: str) -> tuple[State, Dict[str, Any]]:
    with np.load(path) as data:
        meta = json.loads(str(data['meta']))
        fields = {k.removeprefix('field_'): data[k] for k in data.files if k.startswith('field_')}
        st = State(nx=meta.get('nx'), ny=meta.get('ny'), fields={k: v.copy() for k, v in fields.items()})
    return st, meta

__all__ = ["save_checkpoint", "load_checkpoint"]