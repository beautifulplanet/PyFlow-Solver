from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, Any, List

try:  # optional radon
    from radon.complexity import cc_visit  # type: ignore
except Exception:  # pragma: no cover
    cc_visit = None  # type: ignore

STALE_DAYS_DEFAULT = 30


def _collect_functions(root: Path) -> List[Dict[str, Any]]:
    items = []
    for py in root.rglob('*.py'):
        if py.name.startswith('_'):
            continue
        try:
            mtime = py.stat().st_mtime
        except OSError:  # pragma: no cover
            continue
        if cc_visit:
            try:
                blocks = cc_visit(py.read_text(encoding='utf-8'))
                for b in blocks:
                    items.append({
                        'file': str(py.relative_to(root)),
                        'name': b.name,
                        'complexity': b.complexity,
                        'mtime': mtime,
                    })
            except Exception:  # pragma: no cover
                pass
    return items


def _load_baseline(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'avg_complexity': None, 'timestamp': time.time()}
    return json.loads(path.read_text(encoding='utf-8'))


def run_drift_check(project_root: str | Path, stale_days: int = STALE_DAYS_DEFAULT) -> Dict[str, Any]:
    root = Path(project_root)
    knowledge = root / 'knowledge_db'
    baseline_path = knowledge / 'complexity_baseline.json'
    baseline = _load_baseline(baseline_path)
    funcs = _collect_functions(root / 'pyfoamclone')
    now = time.time()
    stale_threshold = now - stale_days * 86400
    stale = [f for f in funcs if f['mtime'] < stale_threshold]
    avg_complexity = sum(f['complexity'] for f in funcs) / len(funcs) if funcs else None
    base_avg = baseline.get('avg_complexity')
    delta = None
    if avg_complexity is not None and base_avg is not None:
        delta = avg_complexity - base_avg
    report = {
        'functions_scanned': len(funcs),
        'stale_functions': stale,
        'avg_complexity': avg_complexity,
        'baseline_avg_complexity': base_avg,
        'complexity_delta': delta,
    }
    return report


def write_baseline(project_root: str | Path) -> Dict[str, Any]:
    root = Path(project_root)
    report = run_drift_check(root, stale_days=100000)  # effectively ignore staleness
    knowledge = root / 'knowledge_db'
    knowledge.mkdir(exist_ok=True)
    baseline_path = knowledge / 'complexity_baseline.json'
    baseline_path.write_text(json.dumps({'avg_complexity': report['avg_complexity'], 'timestamp': time.time()}, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':  # pragma: no cover
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true', help='rewrite baseline complexity snapshot')
    p.add_argument('--project', default='.')
    p.add_argument('--stale-days', type=int, default=STALE_DAYS_DEFAULT)
    args = p.parse_args()
    if args.baseline:
        rep = write_baseline(args.project)
    else:
        rep = run_drift_check(args.project, args.stale_days)
    json.dump(rep, sys.stdout, indent=2)
    print()
