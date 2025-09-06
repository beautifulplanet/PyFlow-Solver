from __future__ import annotations
"""Structured logging utilities (JSONL)."""
import json, os, time, threading, weakref
from typing import Any, Dict

_lock = threading.Lock()
_active = weakref.WeakSet()

class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._path = path
        self.f = open(path, 'a', encoding='utf-8', buffering=1)
        _active.add(self)
    def log(self, record: Dict[str, Any]):
        with _lock:
            record.setdefault('ts', time.time())
            self.f.write(json.dumps(record, separators=(',', ':')) + '\n')
            self.f.flush()
    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
    def __del__(self):  # best effort
        try:
            self.close()
        except Exception:
            pass

def close_all_jsonl_logs():
    for logger in list(_active):
        try:
            logger.close()
        except Exception:
            pass

__all__ = ["JsonlLogger", "close_all_jsonl_logs"]
