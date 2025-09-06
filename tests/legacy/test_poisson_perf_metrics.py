import json, subprocess, sys, os
PY = sys.executable

def test_poisson_metrics_written(tmp_path):  # LEGACY (will be moved to tests/legacy)
    metrics = tmp_path / 'metrics.json'
    cmd = [PY, 'poisson_manufactured_test.py', '--grids', '17', '33', '65', '--tol', '1e-6', '--metrics-out', str(metrics)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert metrics.exists(), 'metrics file not written'
    data = json.loads(metrics.read_text())
    assert 'grids' in data and len(data['grids']) == 3
    # basic sanity: iterations positive and error norms present
    for g in data['grids']:
        assert g['iterations'] > 10
        assert g['l2_error'] > 0 and g['linf_error'] > 0
    assert 'meta' in data and 'monotonic_iterations' in data['meta']
