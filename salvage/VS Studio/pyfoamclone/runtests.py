from __future__ import annotations
import sys
import pathlib

# Ensure project root on path
root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root))

try:
    import pytest  # type: ignore
except ImportError:  # pragma: no cover
    print("pytest not installed", file=sys.stderr)
    sys.exit(1)

# Default: run entire tests directory quietly; allow extra args passthrough
args = ["-q", "tests"] + sys.argv[1:]
raise SystemExit(pytest.main(args))
