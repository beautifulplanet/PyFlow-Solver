import subprocess, sys, json, os

PY = sys.executable

def test_poisson_ci_gate_pass():  # LEGACY (will be moved to tests/legacy)
    cmd = [PY, 'poisson_manufactured_test.py', '--grids', '17', '33', '65', '--ci-gate', '--tol', '1e-6']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'CI GATE PASS' in proc.stdout


def test_poisson_ci_gate_fail_insufficient_pairs():
    cmd = [PY, 'poisson_manufactured_test.py', '--grids', '17', '--ci-gate', '--tol', '1e-6']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0, 'Expected failure due to insufficient grids'
    assert 'insufficient grids' in proc.stdout
