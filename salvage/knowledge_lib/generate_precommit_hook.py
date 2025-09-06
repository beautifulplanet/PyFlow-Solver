"""Generate a git pre-commit hook script that enforces complexity guardrails.
The hook fails if PRECOMMIT_COMPLEXITY_REPORT.md contains any lines beyond the header.

Usage: python generate_precommit_hook.py
Then copy .generated_hooks/pre-commit to .git/hooks/pre-commit and make it executable.
"""
from __future__ import annotations
from pathlib import Path
import os

ROOT = Path(__file__).parent
OUT_DIR = ROOT/'.generated_hooks'
OUT_DIR.mkdir(exist_ok=True)

def main():
    report = ROOT/'PRECOMMIT_COMPLEXITY_REPORT.md'
    hook = OUT_DIR/'pre-commit'
    script = f"""#!/usr/bin/env python
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
report = root / 'PRECOMMIT_COMPLEXITY_REPORT.md'
if not report.exists():
    sys.exit(0)
lines = report.read_text(encoding='utf-8', errors='ignore').splitlines()
violations = [l for l in lines if l.startswith('- ')]
if violations:
    print('[pre-commit] Complexity violations detected:', len(violations))
    for v in violations[:15]:
        print('  ', v)
    print('Abort commit. Refactor or adjust thresholds.')
    sys.exit(1)
sys.exit(0)
"""
    hook.write_text(script, encoding='utf-8')
    print('Generated hook at', hook)
    print('Copy it into .git/hooks/pre-commit (overwriting if desired).')

if __name__ == '__main__':
    main()
