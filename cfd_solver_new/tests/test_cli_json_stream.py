import json
import sys
import subprocess
import shutil
import os

def test_cli_json_stream_basic():
    exe = sys.executable
    # Run a very short simulation to capture JSON lines
    cmd = [exe, '-m', 'pyflow.cli', '--nx', '8', '--ny', '8', '--steps', '3', '--json-stream', '--disable-advection']
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    # Expect exactly 3 lines
    assert len(lines) == 3, f"Expected 3 JSON lines, got {len(lines)}: {lines}"
    for line in lines:
        data = json.loads(line)
        assert data.get('type') == 'step'
        assert 'iteration' in data and isinstance(data['iteration'], int)
        assert 'dt' in data and data['dt'] > 0
        assert 'residuals' in data and 'continuity' in data['residuals']
        assert 'diagnostics' in data
