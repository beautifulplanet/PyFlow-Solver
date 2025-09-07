import sys, os
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
import pytest
from pyflow.logging.structured import close_all_jsonl_logs

@pytest.fixture(autouse=True)
def _close_jsonl_logs_autouse():
    yield
    close_all_jsonl_logs()
