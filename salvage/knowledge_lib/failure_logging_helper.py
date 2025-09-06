"""Structured failure logging helper.
Usage:
    from failure_logging_helper import log_failure
    try:
        ...
    except Exception as e:
        log_failure('module_or_phase', e, context={'param': val})
"""
from __future__ import annotations
import json, traceback, time, os
from datetime import datetime, timezone
LOG_PATH = os.environ.get('CFD_FAILURE_LOG','failure_events.jsonl')
def log_failure(stage: str, err: Exception, context: dict|None=None):
    rec = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'stage': stage,
        'error_type': type(err).__name__,
        'message': str(err)[:400],
        'context': context or {},
        'traceback': ''.join(traceback.format_exception(type(err), err, err.__traceback__))[-5000:]
    }
    with open(LOG_PATH,'a',encoding='utf-8') as f:
        f.write(json.dumps(rec)+'
')
