from __future__ import annotations

"""Structured logging utilities (JSONL) with optional in-memory stream support."""
import json
import os
import threading
import time
import weakref
from typing import IO, Any

_lock = threading.Lock()
_active = weakref.WeakSet()

class JsonlLogger:
    def __init__(self, path: str | None = None, stream: IO[str] | None = None):
        if path is None and stream is None:
            raise ValueError("JsonlLogger requires either path or stream")
        self._path = path
        self._stream = stream
        self._file = None
        if path is not None:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            # line-buffered text file
            self._file = open(path, 'a', encoding='utf-8', buffering=1)
        _active.add(self)
    def _sink(self):
        return self._stream if self._stream is not None else self._file
    def log(self, record: dict[str, Any]):
        sink = self._sink()
        if sink is None:
            return
        with _lock:
            record.setdefault('ts', time.time())
            sink.write(json.dumps(record, separators=(',', ':')) + '\n')
            try:
                sink.flush()
            except Exception:
                pass
    def close(self):
        if self._file is not None:
            try: self._file.close()
            except Exception: pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()
    def __del__(self):  # best effort
        try: self.close()
        except Exception: pass

def close_all_jsonl_logs():
    for logger in list(_active):
        try: logger.close()
        except Exception: pass

__all__ = ["JsonlLogger", "close_all_jsonl_logs"]
