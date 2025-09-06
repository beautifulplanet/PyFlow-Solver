"""Runtime capture utilities: persist latest pytest results and terminal command transcripts.

Provides two main helpers:
- capture_tests(): run pytest (optionally with args) and write JSON summary to test_run_results.json
- record_command(cmd): append a command and its output to runtime_commands.log for later inspection

These are lightweight and do not alter normal operation; you can import and call them
from notebooks, scripts, or an interactive session, avoiding repeated manual copy/paste.
"""
from __future__ import annotations
import subprocess, json, time, shlex, sys, pathlib
from typing import List, Dict, Any, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_RESULTS = ROOT / "test_run_results.json"
COMMAND_LOG = ROOT / "runtime_commands.log"


def capture_tests(pytest_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute pytest capturing individual test outcomes into JSON.
    Relies on pytest being installed in current environment.
    """
    args = [sys.executable, '-m', 'pytest', str(ROOT / 'tests'), '-q']
    if pytest_args:
        args.extend(pytest_args)
    start = time.time()
    proc = subprocess.run(args, capture_output=True, text=True)
    duration = time.time() - start
    # Fallback parsing: count lines with PASSED / FAILED; detailed node info already stored via test_results_recorder if desired.
    output = proc.stdout + "\n" + proc.stderr
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'exit_code': proc.returncode,
        'duration_sec': duration,
        'raw_output_head': '\n'.join(output.splitlines()[:400]),
    }
    # Keep last N summaries inside JSON file as history list
    existing: Dict[str, Any]
    if TEST_RESULTS.exists():
        try:
            existing = json.loads(TEST_RESULTS.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}
    history = existing.get('history', [])
    history.append(summary)
    existing['history'] = history[-20:]  # cap history
    TEST_RESULTS.write_text(json.dumps(existing, indent=2))
    return summary


def record_command(cmd: str) -> str:
    """Run a shell command, append transcript to runtime_commands.log, return output."""
    start = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    delta = time.time() - start
    entry = [
        f"# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"$ {cmd}",
        proc.stdout.rstrip(),
        ("[stderr]\n" + proc.stderr.rstrip()) if proc.stderr.strip() else "",
        f"[exit_code={proc.returncode} duration={delta:.3f}s]",
        ""  # blank separator
    ]
    with COMMAND_LOG.open('a', encoding='utf-8') as fh:
        fh.write('\n'.join(e for e in entry if e is not None))
    return proc.stdout

if __name__ == '__main__':  # pragma: no cover
    capture_tests()
    print(f"Updated {TEST_RESULTS.name}; command log at {COMMAND_LOG.name}")
