#!/usr/bin/env python
"""Fail CI if unexpected duplicate test collection occurs.

Heuristic: collect nodeids via pytest --collect-only -q and ensure no duplicate
module/function pairs and expected count within tolerance.
"""
from __future__ import annotations
import subprocess, sys, re

def main():
    try:
        out = subprocess.check_output([sys.executable, '-m', 'pytest', '--collect-only', '-q'], text=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    # Lines of form path: count
    dup_paths = []
    seen = set()
    total = 0
    pat = re.compile(r'^(tests/[^:]+):\s+(\d+)$')
    for l in lines:
        m = pat.match(l)
        if not m:
            continue
        path, count = m.group(1), int(m.group(2))
        total += count
        if path in seen:
            dup_paths.append(path)
        else:
            seen.add(path)
    if dup_paths:
        print(f"ERROR: duplicate test modules collected: {dup_paths}")
        sys.exit(1)
    # Expected current fast test count (update if tests added intentionally)
    EXPECTED_MIN = 30
    if total < EXPECTED_MIN:
        print(f"ERROR: too few tests collected ({total}) vs expected minimum {EXPECTED_MIN}")
        sys.exit(1)
    print(f"Test collection OK: {total} tests across {len(seen)} modules")

if __name__ == '__main__':
    main()
