from __future__ import annotations
"""Basic tests for AI control layer parsing and CLI argument translation.

These are intentionally lightweight to avoid long simulations.
"""
import sys, json, subprocess
from pyflow.ai.schema import SimulationRequest
from pyflow.ai.nl_parser import parse_natural_language


def test_parse_basic_grid_re():
    req = parse_natural_language("Run lid cavity at Re=250 on 96x64 grid for 300 steps with CFL=0.4")
    assert req.problem == 'lid_cavity'
    assert req.Re == 250
    assert req.nx == 96 and req.ny == 64
    assert req.steps == 300
    assert abs(req.cfl - 0.4) < 1e-12


def test_parse_scheme_threshold_and_verbose():
    req = parse_natural_language("cavity case quick scheme continuity < 1e-5 verbose")
    assert req.scheme == 'quick'
    assert req.continuity_threshold == 1e-5
    assert req.diagnostics is True


def test_cli_args_translation():
    req = SimulationRequest(nx=32, ny=24, Re=150, lid_velocity=1.2, steps=10, scheme='upwind', cfl=0.3, cfl_growth=1.02, diagnostics=False, json_stream=True)
    args = req.to_cli_args()
    # Ensure essential flags present
    joined = ' '.join(args)
    for flag in ["--nx=32", "--ny=24", "--re=150", "--lid-velocity=1.2", "--steps=10", "--scheme=upwind", "--cfl=0.3"]:
        assert flag in joined


def test_end_to_end_short_run():
    # Very short run to confirm subprocess launch path works
    req = SimulationRequest(nx=16, ny=16, steps=2, json_stream=True)
    cmd = [sys.executable, '-m', 'pyflow.cli', *req.to_cli_args()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    # Expect at least 2 JSON lines
    assert len(lines) >= 2
    first = json.loads(lines[0])
    assert first.get('type') == 'step'
