from pyfoamclone.tools.drift_check import run_drift_check
from pathlib import Path


def test_drift_check_structure(tmp_path: Path):
    # Use repo root
    from pyfoamclone import __file__ as init_file
    project_root = Path(init_file).parent.parent
    rep = run_drift_check(project_root, stale_days=0)  # stale_days=0 marks all as stale
    assert 'functions_scanned' in rep and isinstance(rep['functions_scanned'], int)
    assert 'stale_functions' in rep and isinstance(rep['stale_functions'], list)
    assert 'avg_complexity' in rep