import pytest

@pytest.fixture(autouse=True)
def _close_jsonl_logs_autouse():
    yield
    # TODO: Add close_all_jsonl_logs() when implemented
