import pytest
from pyflow.logging.structured import close_all_jsonl_logs

@pytest.fixture(autouse=True)
def _close_jsonl_logs_autouse():
    yield
    close_all_jsonl_logs()
