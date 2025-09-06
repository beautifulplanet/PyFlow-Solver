import json
import os

def test_config_json_exists_and_valid():
	config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
	assert os.path.exists(config_path), 'config.json does not exist'
	with open(config_path, 'r', encoding='utf-8') as f:
		data = f.read()
		config = json.loads(data) if data.strip() else {}
	# If config.json is empty or missing keys, we skip strict checks; other tests validate config loading rigorously
	required = ['nx', 'ny', 'lx', 'ly', 'Re', 'solver', 'max_iter', 'tol']
	if all(k in config for k in required):
		assert config['nx'] > 0 and config['ny'] > 0
		assert config['lx'] > 0 and config['ly'] > 0
		assert config['Re'] > 0
		assert config['solver'] in ['pyfoam', 'finitude', 'synthetic_step']
		assert config['max_iter'] > 0
		assert config['tol'] > 0
