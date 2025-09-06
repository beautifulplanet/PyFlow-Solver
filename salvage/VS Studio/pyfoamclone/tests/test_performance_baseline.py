import time
import pytest
from pathlib import Path
from pyfoamclone.scripts.run_case import run


@pytest.mark.perf
def test_performance_small_case():
    case = Path(__file__).parent.parent / 'cases' / 'lid_cavity_re100.json'
    start = time.time()
    run(str(case))
    elapsed = time.time() - start
    assert elapsed < 0.5  # baseline gate
