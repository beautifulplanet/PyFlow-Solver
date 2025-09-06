from __future__ import annotations
"""Structured logging utilities (JSONL).

Lightweight wrapper writing one JSON object per line for downstream ingestion.
Allows tagging events (step, checkpoint, error) and ensures flush on each line.
"""
import json, os, time, threading
from typing import Any, Dict, Optional

_lock = threading.Lock()

class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8', buffering=1)
    def log(self, record: Dict[str, Any]):
        with _lock:
            record.setdefault('ts', time.time())
            self.f.write(json.dumps(record, separators=(',', ':')) + '\n')
            self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass

__all__ = ["JsonlLogger"]