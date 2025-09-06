"""Generate a richly structured learning dataset from the entire workspace.

Goal:
  Convert raw historical artifacts (code, notebooks, logs, prototypes, fail logs, finals) into a
  machine-consumable supervision corpus that captures:
    - What worked (success patterns)
    - What failed (anti-patterns & root causes where detectable heuristically)
    - Evolution / lifecycle stage (prototype -> iteration -> final)
    - Lightweight static code metrics
    - Heuristic remediation guidance

Outputs:
  1. learning_dataset.jsonl  (one JSON object per file)
  2. learning_dataset_summary.json (aggregate statistics & taxonomies)
  3. learning_report.md (human-readable deep-dive)
  4. ai_training_prompt.txt (prompt template for fine-tuning / RAG assistants)

Heuristics only (no external LLM calls) so it is deterministic & reproducible offline.
"""

from __future__ import annotations

import os
import re
import json
import math
import ast
import warnings
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Iterable

ROOT = Path(__file__).parent
MAX_READ = 1_000_000  # 1 MB safety cap per file
TEXT_EXT = {'.txt', '.md', '.rst', '.log'}
CODE_EXT = {'.py', '.c', '.cpp', '.h', '.hpp', '.js', '.ts'}
NOTEBOOK_EXT = {'.ipynb'}
BINARY_HINT = {'.png', '.jpg', '.jpeg', '.gif', '.zip', '.exe', '.pdf', '.dll'}
EXCLUDED_DIR_NAMES = {'.venv','venv','env','site-packages','dist-packages','__pycache__','build','egg-info','node_modules','.git'}

SUCCESS_TOKENS = [
    'success', 'final', 'finale', 'triumph', 'complete', 'done', 'passed', 'working'
]
FAIL_TOKENS = [
    'fail', 'failure', 'error', 'traceback', 'bug', 'broken', 'wonky', 'issue'
]
PROTOTYPE_TOKENS = [
    'proto', 'prototype', 'draft', 'v0', 'version_0', 'experiment', 'exp'
]
LEGACY_TOKENS = ['legacy', 'old', 'archive']
LOG_TOKENS = ['log', 'logs', 'journal']

KEY_KWS = ['CFD', 'cfd', 'solver', 'navier', 'stokes', 'mesh', 'fluid', 'ai', 'copilot', 'prompt']

def evolution_tag(path: str) -> str:
    name = Path(path).name.lower()
    if any(tok in name for tok in ['proto','draft','v0','experiment']):
        return 'prototype'
    if any(tok in name for tok in ['v1','v2','iter','beta','validate','validation']):
        return 'validated'
    if any(tok in name for tok in ['final','stable','release','prod']):
        return 'stable'
    return 'unspecified'

RE_MEDIATION_PATTERNS = [
    (re.compile(r"missing (module|file|dependency)", re.I), 'Add required dependency & pin version.'),
    (re.compile(r"syntaxerror|indentationerror", re.I), 'Run linter/formatter; add tests for parsing.'),
    (re.compile(r"indexerror", re.I), 'Validate array bounds before access.'),
    (re.compile(r"keyerror", re.I), 'Use dict.get() or validate keys prior.'),
    (re.compile(r"memory", re.I), 'Consider streaming or chunked processing.'),
    (re.compile(r"timeout|took too long", re.I), 'Profile hotspots; add timeouts.'),
]


def hash_path(path: Path) -> str:
    return hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]


def safe_read(path: Path) -> str:
    try:
        with open(path, 'rb') as f:
            raw = f.read(MAX_READ + 1)
        if any(b'\x00' in raw[i:i+100] for i in range(0, min(len(raw), 2000), 100)):
            return ''  # likely binary
        text = raw[:MAX_READ].decode('utf-8', errors='replace')
        return text
    except Exception:
        return ''


def classify_from_path(path: Path) -> List[str]:
    p = str(path).lower()
    labels = []
    if any(tok in p for tok in SUCCESS_TOKENS):
        labels.append('success-indicator')
    if any(tok in p for tok in FAIL_TOKENS):
        labels.append('failure-indicator')
    if any(tok in p for tok in PROTOTYPE_TOKENS):
        labels.append('prototype')
    if any(tok in p for tok in LEGACY_TOKENS):
        labels.append('legacy')
    if any(tok in p for tok in LOG_TOKENS):
        labels.append('log')
    if not labels:
        labels.append('uncategorized')
    return labels


def extract_datetime_hint(name: str) -> str | None:
    # Match patterns like 250803_231838 or 20250829 etc.
    m = re.search(r'(\d{6}[_-]?\d{6})', name)  # yymmdd_hhmmss
    if m:
        token = m.group(1).replace('-', '_')
        try:
            dt = datetime.strptime(token, '%y%m%d_%H%M%S')
            return dt.isoformat()
        except ValueError:
            pass
    m2 = re.search(r'(20\d{6})', name)  # yyyymmdd
    if m2:
        try:
            dt = datetime.strptime(m2.group(1), '%Y%m%d')
            return dt.isoformat()
        except ValueError:
            pass
    return None


def analyze_code(text: str, filename: str) -> Dict[str, Any]:
    metrics = {
        'functions': 0,
        'classes': 0,
        'imports': 0,
        'external_imports': [],
        'avg_function_length': None,
        'syntax_warnings': 0,
    }
    if not text.strip():
        return metrics
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always', SyntaxWarning)
            tree = ast.parse(text, filename=filename)
        metrics['syntax_warnings'] = sum(1 for w in wlist if issubclass(w.category, SyntaxWarning))
    except Exception:
        # fallback regex counts
        metrics['functions'] = text.count('\ndef ') + text.startswith('def ')
        metrics['classes'] = text.count('\nclass ') + text.startswith('class ')
        metrics['imports'] = text.count('\nimport ') + text.count('\nfrom ')
        return metrics
    func_lengths = []
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics['functions'] += 1
            if getattr(node, 'body', None):
                start = getattr(node.body[0], 'lineno', node.lineno)
                end = getattr(node.body[-1], 'end_lineno', node.body[-1].lineno)
                func_lengths.append(max(0, end - start + 1))
        elif isinstance(node, ast.ClassDef):
            metrics['classes'] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            metrics['imports'] += 1
            names = [alias.name.split('.')[0] for alias in node.names]
            for n in names:
                if n not in {'os', 'sys', 're', 'json', 'math', 'typing', 'pathlib', 'datetime', 'collections', 'hashlib', 'ast'}:
                    imports.add(n)
    if func_lengths:
        metrics['avg_function_length'] = sum(func_lengths) / len(func_lengths)
    metrics['external_imports'] = sorted(imports)
    return metrics


def extract_fail_reason(text: str) -> List[str]:
    lines = text.splitlines()
    reasons = []
    for line in lines[:400]:  # only first part for speed
        low = line.lower()
        if any(tok in low for tok in FAIL_TOKENS):
            snippet = line.strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + '...'
            reasons.append(snippet)
    return reasons[:15]


def remediation_suggestions(text: str) -> List[str]:
    sugs = []
    for pat, rec in RE_MEDIATION_PATTERNS:
        if pat.search(text):
            sugs.append(rec)
    return list(dict.fromkeys(sugs))[:10]


def keyword_hits(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for k in KEY_KWS:
        if k.lower() in lower:
            found.append(k)
    return found


def size_bucket(n: int) -> str:
    if n < 1_000: return '<1KB'
    if n < 10_000: return '1-10KB'
    if n < 100_000: return '10-100KB'
    if n < 1_000_000: return '100KB-1MB'
    return '>=1MB'


def build_entry(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    size = path.stat().st_size
    raw = '' if ext in BINARY_HINT else safe_read(path)
    labels = classify_from_path(path)
    dt_hint = extract_datetime_hint(path.name)
    entry: Dict[str, Any] = {
        'id': hash_path(path),
        'path': str(path),
        'ext': ext or 'NONE',
        'size': size,
        'size_bucket': size_bucket(size),
        'labels': labels,
        'datetime_hint': dt_hint,
    'evolution_tag': evolution_tag(str(path)),
        'keyword_hits': keyword_hits(raw) if raw else [],
        'fail_reasons': extract_fail_reason(raw) if 'failure-indicator' in labels or 'log' in labels else [],
        'remediation': remediation_suggestions(raw) if raw else [],
        'code_metrics': None,
        'notebook_meta': None,
        'text_stats': None,
    }
    if ext in CODE_EXT:
        entry['code_metrics'] = analyze_code(raw, str(path))
    elif ext in NOTEBOOK_EXT and raw:
        try:
            import nbformat  # type: ignore
            nb = nbformat.reads(raw, as_version=4)
            code_cells = sum(1 for c in nb.cells if c.get('cell_type') == 'code')
            md_cells = sum(1 for c in nb.cells if c.get('cell_type') == 'markdown')
            entry['notebook_meta'] = {
                'cells': len(nb.cells),
                'code_cells': code_cells,
                'markdown_cells': md_cells,
            }
        except Exception as e:
            entry['notebook_meta'] = {'error': str(e)}
    elif ext in TEXT_EXT or ext == '.':
        if raw:
            lines = raw.splitlines()
            entry['text_stats'] = {
                'lines': len(lines),
                'avg_line_length': round(sum(len(l) for l in lines)/len(lines), 2) if lines else 0,
            }
    return entry


def aggregate(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        'total_entries': len(entries),
        'by_ext': {},
        'label_counts': {},
        'size_buckets': {},
        'avg_functions_per_code_file': 0.0,
        'avg_classes_per_code_file': 0.0,
        'top_external_imports': [],
        'failure_rate': 0.0,
    }
    code_files = 0
    total_funcs = 0
    total_classes = 0
    external_import_freq = {}
    failures = 0
    for e in entries:
        agg['by_ext'][e['ext']] = agg['by_ext'].get(e['ext'], 0) + 1
        for lab in e['labels']:
            agg['label_counts'][lab] = agg['label_counts'].get(lab, 0) + 1
        agg['size_buckets'][e['size_bucket']] = agg['size_buckets'].get(e['size_bucket'], 0) + 1
        if e['code_metrics']:
            code_files += 1
            total_funcs += e['code_metrics']['functions']
            total_classes += e['code_metrics']['classes']
            for imp in e['code_metrics']['external_imports']:
                external_import_freq[imp] = external_import_freq.get(imp, 0) + 1
        if 'failure-indicator' in e['labels']:
            failures += 1
    if code_files:
        agg['avg_functions_per_code_file'] = round(total_funcs / code_files, 2)
        agg['avg_classes_per_code_file'] = round(total_classes / code_files, 2)
    agg['top_external_imports'] = sorted(external_import_freq.items(), key=lambda x: -x[1])[:25]
    if entries:
        agg['failure_rate'] = round(failures / len(entries), 4)
    return agg


def make_report(entries: List[Dict[str, Any]], agg: Dict[str, Any]) -> str:
    lines = []
    lines.append('# Workspace Learning Report')
    lines.append('Generated: ' + datetime.now(timezone.utc).isoformat())
    lines.append('')
    lines.append('## Aggregate Metrics')
    lines.append('Total files analyzed: **{}**'.format(agg['total_entries']))
    lines.append('Failure-indicator files: {} ({}%)'.format(
        agg['label_counts'].get('failure-indicator', 0),
        round(100 * agg['failure_rate'], 2)))
    lines.append('Average functions per code file: {}'.format(agg['avg_functions_per_code_file']))
    lines.append('Average classes per code file: {}'.format(agg['avg_classes_per_code_file']))
    lines.append('Top external imports: ' + ', '.join(f"{k}:{v}" for k, v in agg['top_external_imports']))
    lines.append('')
    def section(title: str):
        lines.append('\n## ' + title)
    section('Label Distribution')
    for lab, cnt in sorted(agg['label_counts'].items(), key=lambda x: -x[1]):
        lines.append(f"- {lab}: {cnt}")
    section('Representative Failures (first 10)')
    failures = [e for e in entries if 'failure-indicator' in e['labels']][:10]
    for f in failures:
        fr = f['fail_reasons'][:2]
        lines.append(f"- {f['path']} | reasons: {fr}")
    section('Success Indicators (first 10)')
    successes = [e for e in entries if 'success-indicator' in e['labels']][:10]
    for s in successes:
        lines.append(f"- {s['path']} | keywords: {s['keyword_hits'][:5]}")
    section('Heuristic Lessons')
    lines.append('- Prefer small, composable functions (avg length evaluated).')
    lines.append('- Capture failure context early in log naming conventions.')
    lines.append('- External import diversity hints at experimentation; curate a stable core set.')
    lines.append('- Prototype to final path can be traced via name evolution (v0 -> final).')
    section('Data Schema Summary')
    lines.append('Each JSONL line fields: id, path, ext, size, size_bucket, labels[], datetime_hint, keyword_hits[], fail_reasons[], remediation[], code_metrics?, notebook_meta?, text_stats?')
    return '\n'.join(lines)


def make_training_prompt(agg: Dict[str, Any]) -> str:
        top_imports = ', '.join([k for k,_ in agg['top_external_imports'][:8]])
        prompt = rf"""You are an AI assistant being fine-tuned on a historical CFD + AI code generation corpus.
The dataset supplies per-file structured metadata capturing successes, failures, prototypes, and resolutions.
Use it to:
1. Recommend reuse of SUCCESS patterns (label success-indicator) when similar intent appears.
2. Detect anti-patterns from failure-indicator + fail_reasons and propose remediations.
3. Infer lifecycle stage (prototype -> final) and advise next maturation steps.
4. Prioritize stability: reference external imports only if frequently used (top imports: {top_imports}).
5. Always surface at least one preventative measure learned from prior failures when generating new CFD solver code.

Output format guidelines for future queries:
{{
    "recommendations": ["actionable step 1", "actionable step 2"],
    "reused_patterns": ["path/or/identifier"],
    "risk_warnings": ["concise risk"],
    "preventative_tests": ["test idea"],
    "explanations": "Short rationale tying advice to corpus evidence"
}}

If user supplies new code, map its characteristics (function count, imports, keywords) to nearest prior examples by shared labels or imports.
Reject requests to reproduce raw proprietary text; only summarize patterns.
"""
        return prompt


def main() -> None:
    # Suppress SyntaxWarning noise globally
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    all_paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        for fname in filenames:
            all_paths.append(Path(dirpath) / fname)
    entries: List[Dict[str, Any]] = []
    excluded = 0
    for p in all_paths:
        if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
            excluded += 1
            continue
        try:
            entries.append(build_entry(p))
        except Exception as e:
            entries.append({'id': hash_path(p), 'path': str(p), 'error': str(e), 'labels': ['error']})
    agg = aggregate(entries)

    # Write JSONL
    with open(ROOT / 'learning_dataset.jsonl', 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    # Write summary
    with open(ROOT / 'learning_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)
    # Report
    with open(ROOT / 'learning_report.md', 'w', encoding='utf-8') as f:
        f.write(make_report(entries, agg))
    # Training prompt
    with open(ROOT / 'ai_training_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(make_training_prompt(agg))
    print('Learning dataset generated:')
    print(' - learning_dataset.jsonl')
    print(' - learning_dataset_summary.json')
    print(' - learning_report.md')
    print(' - ai_training_prompt.txt')
    print('Total entries:', len(entries))
    print('Excluded files (env/third-party):', excluded)


if __name__ == '__main__':  # pragma: no cover
    main()
