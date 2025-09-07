import os, sys, pytest
# Ensure this project's src has precedence when multiple 'pyflow' packages exist in workspace
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:  # idempotent
    sys.path.insert(0, _SRC)
from pyflow.logging.structured import close_all_jsonl_logs

@pytest.fixture(autouse=True)
def _close_jsonl_logs_autouse():
    yield
    close_all_jsonl_logs()
