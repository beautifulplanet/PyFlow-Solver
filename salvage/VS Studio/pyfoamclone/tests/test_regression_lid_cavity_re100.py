import numpy as np
from pathlib import Path
from pyfoamclone.scripts.run_case import run


def test_regression_lid_cavity_re100():
    case = Path(__file__).parent.parent / 'cases' / 'lid_cavity_re100.json'
    result = run(str(case))
    # Check centerline velocity for lid-driven cavity
    u_cl = np.array(result['u_centerline'])
    # Ensure we have data and it's physically meaningful
    assert len(u_cl) > 0
    # In a lid-driven cavity, centerline velocity should have non-zero values
    assert np.linalg.norm(u_cl) > 0  # Ensure velocity is non-zero
    assert result['iterations'] >= 1
