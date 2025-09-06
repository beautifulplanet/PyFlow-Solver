import json
from pathlib import Path
import pytest
from pyfoamclone.logging.failure_logging_helper import log_failure


def test_log_failure_writes(tmp_path: Path, capsys):
    p = tmp_path / "fail.jsonl"
    try:
        raise ValueError("boom")
    except Exception as e:  # noqa: PERF203
        log_failure("unit-test", e, {"k": 1}, path=str(p))
    assert p.exists()
    line = p.read_text().strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["stage"] == "unit-test" and rec["error_type"] == "ValueError"
