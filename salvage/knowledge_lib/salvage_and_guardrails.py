"""Salvage high-quality functions from failing files, add evolution tags, and provide guardrails.

Outputs:
  - orphan_salvage.jsonl : reusable high-quality functions inside failure-indicator files
  - evolution_tag_summary.json : mapping of file -> evolution_tag (prototype|validated|stable)
  - failure_logging_helper.py : helper module emitted (idempotent) with structured log API
  - PRECOMMIT_COMPLEXITY_REPORT.md : report of oversized functions

Guardrails:
  - Complexity threshold (default 15) and length threshold (80 lines) for solver functions.
  - Diff guard (optional): if GIT is present, flag functions whose added length > threshold.
"""
from __future__ import annotations
import json, csv, os, ast, hashlib, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).parent
LEARNING_JSONL = ROOT / 'learning_dataset.jsonl'
FUNC_CATALOG = ROOT / 'function_catalog.csv'
FAIL_LABEL = 'failure-indicator'

COMPLEXITY_THRESHOLD = 15
LENGTH_THRESHOLD = 80

def read_jsonl(p: Path):
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        line=line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: pass
    return out

def read_func_catalog(p: Path):
    rows=[]
    if not p.exists(): return rows
    with p.open('r', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                row['length']=int(row['length']); row['complexity']=int(row['complexity'])
                row['doc_present']=row['doc_present'].strip().lower()=='true'
                row['cfd_tags']=[t for t in row['cfd_tags'].split('|') if t]
                row['parameters']=[t for t in row['parameters'].split('|') if t]
            except Exception: pass
            rows.append(row)
    return rows

def build_failure_file_set(entries):
    return {e['path'] for e in entries if FAIL_LABEL in e.get('labels', [])}

def quality_score(f):
    # replicate logic (approx) used earlier; keep in sync if updated
    base=100
    c=f.get('complexity',0); l=f.get('length',0)
    complexity_pen = max(0, c-5)*2
    length_pen = max(0, l-60)*0.5 if l>60 else 0
    doc_bonus = 8 if f.get('doc_present') else -6
    tag_bonus = min(len(f.get('cfd_tags',[]))*2,10)
    param_bonus = 2 if 0 < len(f.get('parameters',[])) <=5 else (0 if len(f.get('parameters',[]))==0 else -2)
    return max(0, round(base - complexity_pen - length_pen + doc_bonus + tag_bonus + param_bonus,2))

def salvage(functions, failing_files):
    salvaged=[]
    for f in functions:
        if f['file'] in failing_files:
            f['quality_score']=quality_score(f)
            if f['quality_score']>=80:
                salvaged.append(f)
    salvaged.sort(key=lambda x: -x['quality_score'])
    return salvaged

def evolution_tag(file_path: str) -> str:
    name = Path(file_path).name.lower()
    if any(tok in name for tok in ['proto','draft','v0','experiment']):
        return 'prototype'
    if any(tok in name for tok in ['v1','v2','iter','beta','validate','validation']):
        return 'validated'
    if any(tok in name for tok in ['final','stable','release','prod']):
        return 'stable'
    return 'unspecified'

def build_evolution_tags(entries):
    mapping={}
    for e in entries:
        mapping[e['path']] = evolution_tag(e['path'])
    return mapping

def ensure_failure_logging_helper():
    helper = ROOT/'failure_logging_helper.py'
    if helper.exists():
        return helper
    helper.write_text('''"""Structured failure logging helper.
Usage:
    from failure_logging_helper import log_failure
    try:
        ...
    except Exception as e:
        log_failure('module_or_phase', e, context={'param': val})
"""
from __future__ import annotations
import json, traceback, time, os
from datetime import datetime, timezone
LOG_PATH = os.environ.get('CFD_FAILURE_LOG','failure_events.jsonl')
def log_failure(stage: str, err: Exception, context: dict|None=None):
    rec = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'stage': stage,
        'error_type': type(err).__name__,
        'message': str(err)[:400],
        'context': context or {},
        'traceback': ''.join(traceback.format_exception(type(err), err, err.__traceback__))[-5000:]
    }
    with open(LOG_PATH,'a',encoding='utf-8') as f:
        f.write(json.dumps(rec)+'\n')
''', encoding='utf-8')
    return helper

def parse_git_diff_added_lines():
    if shutil.which('git') is None:
        return {}
    try:
        out = subprocess.check_output(['git','diff','--unified=0','HEAD'], cwd=ROOT, text=True, errors='ignore')
    except Exception:
        return {}
    added=defaultdict(int)
    current=None
    for line in out.splitlines():
        if line.startswith('+++ b/'):
            current = line[6:]
        elif line.startswith('+') and not line.startswith('+++'):
            if current:
                added[current]+=1
    return added

def oversize_report(functions):
    lines=['# Pre-commit Complexity Report', f'Generated: {datetime.now(timezone.utc).isoformat()}', '', f'Thresholds: complexity>{COMPLEXITY_THRESHOLD} or length>{LENGTH_THRESHOLD} (with complexity>10)']
    added = parse_git_diff_added_lines()
    rows=[]
    for f in functions:
        c=f['complexity']; l=f['length']
        if c>COMPLEXITY_THRESHOLD or (l> LENGTH_THRESHOLD and c>10):
            rows.append((c,l,f['file'],f['name']))
    rows.sort(key=lambda x: (-x[0], -x[1]))
    for c,l,file,name in rows[:200]:
        extra=''
        rel = file.replace(str(ROOT)+os.sep,'')
        if rel in added:
            extra=f' (+{added[rel]} new lines)'
        lines.append(f'- {rel}::{name} complexity={c} length={l}{extra}')
    (ROOT/'PRECOMMIT_COMPLEXITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main():
    entries = read_jsonl(LEARNING_JSONL)
    funcs = read_func_catalog(FUNC_CATALOG)
    failing = build_failure_file_set(entries)
    salvaged = salvage(funcs, failing)
    with open(ROOT/'orphan_salvage.jsonl','w', encoding='utf-8') as f:
        for s in salvaged:
            f.write(json.dumps(s)+'\n')
    evo_tags = build_evolution_tags(entries)
    with open(ROOT/'evolution_tag_summary.json','w', encoding='utf-8') as f:
        json.dump(evo_tags, f, indent=2)
    ensure_failure_logging_helper()
    oversize_report(funcs)
    print('Salvage + guardrail artifacts generated:')
    for p in ['orphan_salvage.jsonl','evolution_tag_summary.json','failure_logging_helper.py','PRECOMMIT_COMPLEXITY_REPORT.md']:
        print(' -', p)
    print('Salvaged functions:', len(salvaged))

if __name__ == '__main__':
    main()
