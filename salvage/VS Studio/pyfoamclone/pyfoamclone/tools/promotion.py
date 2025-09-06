from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

try:  # optional
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover
    cc_visit = None  # type: ignore


@dataclass(slots=True)
class PromotionReport:
    module: str
    doc_coverage: float
    functions: int
    avg_complexity: float | None
    complexity_violations: int
    status: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'module': self.module,
            'doc_coverage': self.doc_coverage,
            'functions': self.functions,
            'avg_complexity': self.avg_complexity,
            'complexity_violations': self.complexity_violations,
            'status': self.status,
            'notes': self.notes,
        }


def analyze_module(path: Path, max_complexity: int = 12) -> PromotionReport:
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    documented = 0
    for f in funcs:
        if ast.get_docstring(f):
            documented += 1
    doc_cov = (documented / len(funcs)) * 100 if funcs else 100.0
    avg_complexity = None
    violations = 0
    notes: List[str] = []
    if cc_visit:
        try:
            blocks = cc_visit(src)
            if blocks:
                avg_complexity = sum(b.complexity for b in blocks) / len(blocks)
                violations = sum(1 for b in blocks if b.complexity > max_complexity)
        except Exception as e:  # pragma: no cover
            notes.append(f"complexity_parse_error: {e}")
    status = 'pass' if doc_cov == 100.0 and violations == 0 else 'fail'
    if doc_cov < 100:
        notes.append('incomplete_docstrings')
    if violations:
        notes.append('complexity_violations')
    return PromotionReport(path.name, doc_cov, len(funcs), avg_complexity, violations, status, notes)


def promote(path: str | Path, apply: bool = False) -> PromotionReport:
    path = Path(path)
    report = analyze_module(path)
    if report.status == 'pass' and apply:
        target_dir = Path.cwd() / 'cfd_core'
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / path.name.replace('_proto', '')
        target_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
    return report


if __name__ == '__main__':  # pragma: no cover
    import sys
    rep = promote(sys.argv[1], apply='--apply' in sys.argv)
    print(json.dumps(rep.to_dict(), indent=2))
