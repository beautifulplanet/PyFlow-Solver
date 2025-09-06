"""Utility to run the full test suite programmatically and store structured results.

Invoke (python -m pyfoamclone.tests.test_results_recorder) or import and call
`run_and_store_results()` from elsewhere. This creates/overwrites a JSON file
`test_run_results.json` at repo root with per-test outcome data.

This is optional and does not affect normal pytest runs (pytest auto-discovers
files named test_*.py; to avoid double execution we guard main section).
"""
from __future__ import annotations
import json, time, os, pathlib, importlib
from typing import List, Dict, Any

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None  # noqa

RESULTS_FILENAME = "test_run_results.json"


def run_and_store_results(pytest_args: List[str] | None = None) -> Dict[str, Any]:
    if pytest is None:
        raise RuntimeError("pytest not available")
    # Determine repository root: this file lives in <root>/tests/
    root = pathlib.Path(__file__).resolve().parent.parent
    out_path = root / RESULTS_FILENAME
    # Default: run entire suite quietly with json reporting (requires builtin json report plugin fallback below)
    # We'll implement a lightweight reporter plugin capturing results.
    collected: Dict[str, Dict[str, Any]] = {}

    class CapturePlugin:
        def pytest_runtest_logreport(self, report):  # noqa
            if report.when == "call":
                nodeid = report.nodeid
                collected[nodeid] = {
                    "outcome": report.outcome,
                    "duration": getattr(report, 'duration', None),
                    "longrepr": str(report.longrepr) if report.failed else None,
                }

    tests_dir = root / 'tests'
    args = [str(tests_dir)]
    if pytest_args:
        args.extend(pytest_args)
    start = time.time()
    exit_code = pytest.main(args, plugins=[CapturePlugin()])
    total_time = time.time() - start

    summary = {
        "exit_code": exit_code,
        "total_time_sec": total_time,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "tests": collected,
        "summary": {
            "passed": sum(1 for v in collected.values() if v['outcome'] == 'passed'),
            "failed": sum(1 for v in collected.values() if v['outcome'] == 'failed'),
            "skipped": sum(1 for v in collected.values() if v['outcome'] == 'skipped'),
            "total": len(collected),
        }
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover
    res = run_and_store_results()
    print(f"Wrote {RESULTS_FILENAME} with {len(res['tests'])} test records (exit={res['exit_code']}).")
