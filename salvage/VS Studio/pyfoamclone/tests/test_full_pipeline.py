import os
import sys
import subprocess
import json


def _read(path: str) -> str:
	with open(path, 'r', encoding='utf-8') as f:
		return f.read()


def _write(path: str, data: str) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		f.write(data)


def _set_solver(config_path: str, solver: str) -> None:
	data = json.loads(_read(config_path) or '{}')
	data['solver'] = solver
	_write(config_path, json.dumps(data))


def test_run_simulation_script_pyfoam():
	script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_simulation.py')
	config = os.path.join(os.path.dirname(__file__), '..', 'config.json')
	assert os.path.isfile(script), 'run_simulation.py not found'
	assert os.path.isfile(config), 'config.json not found'

	original = _read(config)
	try:
		_set_solver(config, 'pyfoam')
		result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
		assert result.returncode == 0, f'PyFOAM run failed: {result.stderr}'
	finally:
		_write(config, original)


def test_run_simulation_script_finitude():
	script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_simulation.py')
	config = os.path.join(os.path.dirname(__file__), '..', 'config.json')
	assert os.path.isfile(script), 'run_simulation.py not found'
	assert os.path.isfile(config), 'config.json not found'

	original = _read(config)
	try:
		_set_solver(config, 'finitude')
		result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
		assert result.returncode == 0, f'Finitude run failed: {result.stderr}'
	finally:
		_write(config, original)
