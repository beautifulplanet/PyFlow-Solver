import os
import importlib.util

def test_chimera_package_importable():
	spec = importlib.util.find_spec('pyfoamclone.chimera')
	assert spec is not None, 'pyfoamclone.chimera package not importable'
def test_solvers_subpackage_importable():
	spec = importlib.util.find_spec('pyfoamclone.chimera.solvers')
	assert spec is not None, 'pyfoamclone.chimera.solvers subpackage not importable'
def test_utils_subpackage_importable():
	spec = importlib.util.find_spec('pyfoamclone.chimera.utils')
	assert spec is not None, 'pyfoamclone.chimera.utils subpackage not importable'
def test_scripts_folder_exists():
	scripts_path = os.path.join(os.path.dirname(__file__), '..', 'pyfoamclone', 'scripts')
	assert os.path.isdir(scripts_path), f'scripts folder does not exist at {scripts_path}'
def test_config_json_exists():
	config_path = os.path.join(os.path.dirname(__file__), '..', 'pyfoamclone', 'config.json')
	assert os.path.isfile(config_path), f'config.json does not exist at {config_path}'
def test_all_init_files_exist():
	base = os.path.join(os.path.dirname(__file__), '..', 'pyfoamclone', 'chimera')
	for sub in ['', 'solvers', 'utils']:
		path = os.path.join(base, sub, '__init__.py')
		assert os.path.isfile(path), f"Missing __init__.py in {sub or 'chimera'} at {path}"
