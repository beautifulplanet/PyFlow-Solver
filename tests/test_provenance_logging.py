import json, subprocess, sys, os, glob
PY = sys.executable

def test_poisson_creates_provenance_logs(tmp_path):
    # Run twice and ensure two distinct run directories created
    env = os.environ.copy()
    env['PROJECT_NAME'] = 'projA'
    cmd = [PY, 'poisson_manufactured_test.py', '--grids', '17', '33']
    for _ in range(2):
        proc = subprocess.run(cmd, cwd='.', env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    # Find run_meta files
    metas = glob.glob('logs/projA/*/run_*/run_meta.json')
    assert len(metas) >= 2
    # Load one and check required keys
    with open(metas[-1], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for k in ['project_name','run_id','timestamp_utc','python_version']:
        assert k in meta
