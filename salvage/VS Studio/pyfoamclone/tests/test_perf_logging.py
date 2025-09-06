import json
import time
from pathlib import Path
from pyfoamclone.tools.perf import timer, timer_ctx


def test_timer_decorator_writes(tmp_path: Path):
    log = tmp_path / 'perf.jsonl'

    @timer('unit-sleep', log_path=str(log))
    def sleepy():
        time.sleep(0.005)

    sleepy()
    assert log.exists()
    line = log.read_text().strip().splitlines()[0]
    rec = json.loads(line)
    assert rec['label'] == 'unit-sleep' and rec['duration_ms'] > 0


def test_timer_ctx_writes(tmp_path: Path):
    log = tmp_path / 'perf.jsonl'
    with timer_ctx('ctx-op', log_path=str(log)):
        time.sleep(0.002)
    rec = json.loads(log.read_text().strip().splitlines()[0])
    assert rec['label'] == 'ctx-op'