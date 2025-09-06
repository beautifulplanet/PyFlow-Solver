import json
import subprocess
import sys
import os
import shlex

# Basic smoke test running the CLI for a few steps in JSON mode and parsing output.

def test_cli_json_smoke(tmp_path):
    # Use python -m pyflow.cli to execute module
    cmd = [sys.executable, '-m', 'pyflow.cli', '--nx', '8', '--ny', '8', '--steps', '3', '--json', '--disable-advection']
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 3, f"Expected 3 JSON lines, got {len(lines)}: {lines}"
    last_iter = -1
    for l in lines:
        data = json.loads(l)
        assert 'iteration' in data and 'continuity' in data
        assert data['iteration'] > last_iter
        last_iter = data['iteration']
