from pyflow.config.model import SimulationConfig, ConfigError
import pytest
from pydantic import ValidationError

def test_config_basic_hash_stable():
    c1 = SimulationConfig(nx=12, ny=10)
    c2 = SimulationConfig(nx=12, ny=10)
    assert c1.config_hash == c2.config_hash
    c3 = SimulationConfig(nx=13, ny=10)
    assert c1.config_hash != c3.config_hash

@pytest.mark.parametrize("field, value", [
    ("nx", 2),
    ("ny", 0),
    ("Re", -5.0),
    ("cfl_target", 0.0),
    ("cfl_target", 2.0),
    ("cfl_growth", 0.9),
    ("cfl_growth", 2.0),
    ("lin_tol", 1e-20),
    ("lin_tol", 1.0),
    ("lin_maxiter", 0),
])
def test_invalid_scalar_ranges(field, value):
    kwargs = dict(nx=12, ny=12)
    kwargs[field] = value
    with pytest.raises((ValueError, ConfigError)):
        SimulationConfig(**kwargs)

def test_mutual_exclusive_logging():
    with pytest.raises((ConfigError, ValidationError)):
        SimulationConfig(nx=12, ny=12, log_path="a.jsonl", log_stream=object())

def test_aspect_ratio_soft_warning():
    c = SimulationConfig(nx=1000, ny=10)
    # Soft warning stored internally (not enforced); just ensure object builds.
    assert c.nx == 1000

def test_disable_advection_allows_scheme():
    c = SimulationConfig(nx=12, ny=12, disable_advection=True, advection_scheme='upwind')
    assert c.disable_advection and c.advection_scheme == 'upwind'
