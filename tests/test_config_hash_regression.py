from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from pyflow.config.model import config_hash, EXCLUDED_RUNTIME_FIELDS, config_core_dict, HASH_LEN, freeze_config
from pyflow.config.model import SimulationConfig

@dataclass
class DummyCfg:
    nx: int = 32
    ny: int = 32
    Re: float = 100.0
    lid_velocity: float = 1.0
    cfl_target: float = 0.5
    cfl_growth: float = 1.05
    lin_tol: float = 1e-10
    lin_maxiter: int = 300
    disable_advection: bool = False


def test_config_hash_stable_under_runtime_field_mutation():
    base = DummyCfg()
    h0 = config_hash(base)
    assert len(h0) == HASH_LEN
    # Mutate every excluded runtime field (inject if not present)
    for f in sorted(EXCLUDED_RUNTIME_FIELDS):
        mutated = deepcopy(base)
        setattr(mutated, f, 123 if f != 'force_quiet' else True)
        assert config_hash(mutated) == h0, f"Runtime field '{f}' changed hash"  # noqa: E501


def test_config_hash_changes_with_semantic_field():
    base = DummyCfg()
    h0 = config_hash(base)
    mutated = deepcopy(base)
    mutated.Re = 150.0
    assert config_hash(mutated) != h0, "Semantic change (Re) did not affect hash"


def test_core_dict_excludes_runtime_fields():
    base = DummyCfg()
    for f in EXCLUDED_RUNTIME_FIELDS:
        setattr(base, f, 42)
    core = config_core_dict(base)
    for f in EXCLUDED_RUNTIME_FIELDS:
        assert f not in core, f"Excluded field '{f}' leaked into core dict"


def test_all_simulation_config_fields_accounted():
    cfg = SimulationConfig(nx=8, ny=8)
    attrs = {k for k in vars(cfg).keys() if not k.startswith('_')}
    core = set(config_core_dict(cfg).keys())
    excluded = EXCLUDED_RUNTIME_FIELDS
    # Every attr must be either in core (semantic) or excluded runtime list
    for a in attrs:
        assert (a in core) or (a in excluded), f"Config field '{a}' is neither hashed nor excluded"
