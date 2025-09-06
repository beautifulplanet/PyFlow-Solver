import json
from pathlib import Path
import pytest

from pyfoamclone.configuration.loader import load_config


def test_config_valid_tmp(tmp_path: Path):
    cfg = {
        "schema_version": 1,
        "nx": 8,
        "ny": 8,
        "lx": 1.0,
        "ly": 1.0,
        "Re": 100,
        "solver": "pyfoam",
        "max_iter": 10,
        "tol": 1e-4,
        "relax_u": 0.7,
        "relax_p": 0.3
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    c = load_config(p)
    assert c.nx == 8 and c.Re == 100


@pytest.mark.parametrize("missing", ["nx", "ny", "Re"])
def test_config_missing_keys(missing: str, tmp_path: Path):
    cfg = {
        "schema_version": 1,
        "nx": 8,
        "ny": 8,
        "lx": 1.0,
        "ly": 1.0,
        "Re": 100,
        "solver": "pyfoam",
        "max_iter": 10,
        "tol": 1e-4,
    }
    cfg.pop(missing)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(Exception):
        load_config(p)
