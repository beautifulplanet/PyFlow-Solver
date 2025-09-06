import pytest


@pytest.fixture
def benchmark():  # Fallback simple benchmark fixture if pytest-benchmark plugin not installed
    def _bench(fn):
        return fn()
    return _bench
