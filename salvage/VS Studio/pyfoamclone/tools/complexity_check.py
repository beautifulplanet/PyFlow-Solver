"""Lightweight complexity guard using radon if available.

Not a strict gate yet; prints metrics and exits 0 always.
Integrate with CI later to enforce thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

THRESH_FUNC_COMPLEXITY = 12


def main() -> int:
    try:
        from radon.complexity import cc_visit  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        print("radon not installed; skipping complexity check")
        return 0
    root = Path(__file__).resolve().parent.parent / "pyfoamclone"
    py_files = list(root.rglob("*.py"))
    violations = 0
    for f in py_files:
        if f.name.endswith("__init__.py"):
            continue
        code = f.read_text(encoding="utf-8")
        try:
            blocks = cc_visit(code)
        except Exception as e:  # pragma: no cover
            print(f"Failed parsing {f}: {e}")
            continue
        for b in blocks:
            if b.complexity > THRESH_FUNC_COMPLEXITY:
                violations += 1
                print(f"VIOLATION {b.name} ({b.complexity}) in {f.relative_to(root)}")
    if violations:
        print(f"Complexity violations: {violations}")
    else:
        print("No complexity violations detected")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
