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
from dataclasses import asdict
import json, os, tempfile, hashlib, io, subprocess
from typing import Any, Dict, cast
import numpy as np

from ..core.ghost_fields import State
from ..config.model import config_hash, HASH_LEN, config_core_dict


def _hash_config(cfg: Any) -> str:  # backwards alias; delegated to config_hash
    try:
        return config_hash(cfg)
    except Exception:
        return "unknown"


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
    git_rev = None
    try:  # best effort; ignore errors (e.g., not a git repo)
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # pragma: no cover
        pass
    meta: Dict[str, Any] = {
        'iteration': iteration,
        'time': sim_time,
        'nx': state.nx,
        'ny': state.ny,
        'config_hash': _hash_config(cfg),
        'field_hashes': field_hashes,
        'seed': getattr(cfg, 'seed', None),
        'schema_version': 2,
    'hash_len': HASH_LEN,
    'git_commit': git_rev,
        'core_config': config_core_dict(cfg),
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


def load_checkpoint(path: str) -> tuple[State, Dict[str, Any]]:
    with np.load(path) as data:
        meta = json.loads(str(data['meta']))
        fields = {k.removeprefix('field_'): data[k] for k in data.files if k.startswith('field_')}
        st = State(nx=meta.get('nx'), ny=meta.get('ny'), fields={k: v.copy() for k, v in fields.items()})
    return st, meta

__all__ = ["save_checkpoint", "load_checkpoint"]

def save_checkpoint_bytes(state: State, iteration: int, sim_time: float, cfg: Any) -> bytes:
    import io
    field_hashes = {k: _hash_array(v) for k,v in state.fields.items()}
    meta = {"iteration": iteration, "time": sim_time, "nx": state.nx, "ny": state.ny, "config_hash": _hash_config(cfg), "field_hashes": field_hashes, "seed": getattr(cfg,"seed",None), "schema_version":1}
    bio = io.BytesIO()
    arrays = {f"field_{k}": v for k,v in state.fields.items()}
    saver = cast(Any, np.savez_compressed)
    saver(bio, meta=json.dumps(meta), **arrays)
    return bio.getvalue()

def load_checkpoint_bytes(buf: bytes) -> tuple[State, Dict[str, Any]]:
    import io
    bio = io.BytesIO(buf)
    with np.load(bio) as data:
        meta = json.loads(str(data['meta']))
        fields = {k.removeprefix('field_'): data[k] for k in data.files if k.startswith('field_')}
        st = State(nx=meta.get('nx'), ny=meta.get('ny'), fields={k: v.copy() for k,v in fields.items()})
    return st, meta
