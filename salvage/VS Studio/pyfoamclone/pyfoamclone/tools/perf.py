from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Any


def _default_log_path() -> Path:
    return Path.cwd() / 'knowledge_db' / 'perf_log.jsonl'


def _write_log(record: dict, log_path: str | None) -> None:
    path = Path(log_path) if log_path else _default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')


@contextmanager
def timer_ctx(label: str, log_path: str | None = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        rec = {
            'ts': datetime.now().isoformat(),
            'label': label,
            'duration_ms': dt,
        }
        _write_log(rec, log_path)


def timer(label: str, log_path: str | None = None):
    def deco(fn: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                rec = {
                    'ts': datetime.now().isoformat(),
                    'label': label,
                    'duration_ms': dt,
                }
                _write_log(rec, log_path)
        return wrapper
    return deco
