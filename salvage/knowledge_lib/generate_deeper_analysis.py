"""Deeper cross-artifact analysis for CFD + AI workspace.

Inputs (expected pre-generated):
  - learning_dataset.jsonl
  - extended_learning_dataset.jsonl
  - dependency_graph.json
  - function_catalog.csv
  - evolution_chains.json

Outputs:
  - deeper_metrics.json            (aggregate structured metrics)
  - function_quality.csv           (per-function scores)
  - module_centrality.json         (import graph centrality & coupling stats)
  - timeline_metrics.json          (time-bucketed success/failure trends)
  - deeper_analysis_report.md      (human-readable narrative)

Focus areas:
  1. Temporal trends of success vs failure.
  2. Function quality scoring (complexity, length, docs, CFD tag richness, parameter clarity).
  3. Orphan salvage candidates: high quality functions inside failure-indicator files.
  4. Import graph central modules & weakly connected leaves (refactor targets).
  5. Evolution chain maturation deltas (function count & average complexity change).
  6. Refactor recommendations (complex, undocumented, long functions) ranked by impact.

Run after the other generators. Pure offline analysis.
"""
from __future__ import annotations
import csv, json, os, math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

ROOT = Path(__file__).parent

# ---------- Helpers ----------

def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def load_function_catalog(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # normalize types
            try:
                r['lineno'] = int(r['lineno'])
                r['end_lineno'] = int(r['end_lineno'])
                r['length'] = int(r['length'])
                r['complexity'] = int(r['complexity'])
                r['doc_present'] = (r['doc_present'].strip().lower() == 'true')
                r['cfd_tags'] = [t for t in r['cfd_tags'].split('|') if t]
                r['parameters'] = [p for p in r['parameters'].split('|') if p]
            except Exception:
                pass
            rows.append(r)
    return rows

# ---------- Temporal Trends ----------

def bucket_time(ts_iso: str | None):
    if not ts_iso:
        return 'unknown'
    try:
        dt = datetime.fromisoformat(ts_iso.replace('Z',''))
        return f"{dt.year}-{dt.month:02d}"
    except Exception:
        return 'unknown'


def build_timeline(entries: list[dict]):
    buckets = defaultdict(lambda: {'success':0,'failure':0,'total':0})
    for e in entries:
        b = bucket_time(e.get('datetime_hint'))
        buckets[b]['total'] += 1
        labs = e.get('labels', [])
        if 'success-indicator' in labs:
            buckets[b]['success'] += 1
        if 'failure-indicator' in labs:
            buckets[b]['failure'] += 1
    # compute rates
    out = []
    for k,v in sorted(buckets.items()):
        total = v['total'] or 1
        v['failure_rate'] = round(v['failure']/total, 4)
        v['success_rate'] = round(v['success']/total, 4)
        v['bucket'] = k
        out.append(v)
    return out

# ---------- Function Quality ----------

def quality_score(row: dict) -> float:
    length = row.get('length',0)
    complexity = row.get('complexity',0)
    doc = row.get('doc_present', False)
    tags = row.get('cfd_tags',[])
    params = row.get('parameters',[])
    # penalties
    complexity_pen = max(0, complexity - 5) * 2  # heavier penalty after baseline complexity 5
    length_pen = 0
    if length > 60:
        length_pen = (length - 60) * 0.5
    param_bonus = 2 if 0 < len(params) <= 5 else (0 if len(params)==0 else -2)
    doc_bonus = 8 if doc else -6
    tag_bonus = min(len(tags)*2, 10)
    base = 100
    score = base - complexity_pen - length_pen + doc_bonus + tag_bonus + param_bonus
    return round(max(0, score),2)


def enrich_quality(functions: list[dict]):
    for f in functions:
        f['quality_score'] = quality_score(f)
    return functions

# ---------- Import Graph Centrality ----------

def load_dependency_graph(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def compute_module_centrality(graph: dict[str, list[str]]):
    indeg = Counter()
    outdeg = {m: len(deps) for m, deps in graph.items()}
    for m,deps in graph.items():
        for d in deps:
            indeg[d]+=1
    modules = set(graph.keys()) | set(indeg.keys())
    central = []
    for m in modules:
        central.append({
            'module': m,
            'in_degree': indeg.get(m,0),
            'out_degree': outdeg.get(m,0),
            'coupling': indeg.get(m,0) + outdeg.get(m,0)
        })
    central.sort(key=lambda x: (-x['coupling'], -x['in_degree']))
    return central

# ---------- Evolution Chain Deltas ----------

def load_evolution_chains(path: Path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []


def file_stats_map(entries: list[dict]):
    mapping = {}
    for e in entries:
        mapping[Path(e['path']).name] = e
    return mapping


def chain_maturation(chains: list[dict], name_map: dict[str, dict]):
    out = []
    for ch in chains:
        seq = ch.get('sequence', [])
        if len(seq) < 2:
            continue
        first = name_map.get(seq[0])
        last = name_map.get(seq[-1])
        if not first or not last:
            continue
        first_funcs = first.get('code_metrics', {}).get('functions',0)
        last_funcs = last.get('code_metrics', {}).get('functions',0)
        first_classes = first.get('code_metrics', {}).get('classes',0)
        last_classes = last.get('code_metrics', {}).get('classes',0)
        maturation = {
            'base': ch.get('base'),
            'length': len(seq),
            'sequence': seq,
            'func_count_delta': last_funcs - first_funcs,
            'class_count_delta': last_classes - first_classes,
        }
        out.append(maturation)
    out.sort(key=lambda x: -x['length'])
    return out

# ---------- Orphan High-Quality Functions ----------

def orphan_salvage(functions: list[dict], entries: list[dict]):
    fail_files = {e['path'] for e in entries if 'failure-indicator' in e.get('labels', [])}
    orphans = [f for f in functions if f.get('quality_score',0) >= 80 and f['file'] in fail_files]
    orphans.sort(key=lambda x: -x['quality_score'])
    return orphans

# ---------- Refactor Candidates ----------

def refactor_candidates(functions: list[dict]):
    cands = []
    for f in functions:
        if f['complexity'] > 15 or (f['length'] > 80 and f['complexity'] > 10) or (not f['doc_present'] and f['complexity'] > 10):
            penalty = (f['complexity'] - 10) * 3 + max(0, f['length'] - 60)
            cands.append({
                'id': f['id'],
                'file': f['file'],
                'name': f['name'],
                'complexity': f['complexity'],
                'length': f['length'],
                'doc_present': f['doc_present'],
                'quality_score': f.get('quality_score',0),
                'priority': penalty
            })
    cands.sort(key=lambda x: (-x['priority'], x['quality_score']))
    return cands[:100]

# ---------- Main ----------

def main():
    learning_entries = load_jsonl(ROOT/'learning_dataset.jsonl')
    func_rows = load_function_catalog(ROOT/'function_catalog.csv')
    func_rows = enrich_quality(func_rows)
    ext_funcs = load_jsonl(ROOT/'extended_learning_dataset.jsonl')  # may contain overlapping info (not used directly)
    dep_graph = load_dependency_graph(ROOT/'dependency_graph.json')
    evo_chains = load_evolution_chains(ROOT/'evolution_chains.json')

    timeline = build_timeline(learning_entries)
    name_map = file_stats_map(learning_entries)
    maturation = chain_maturation(evo_chains, name_map)
    centrality = compute_module_centrality(dep_graph)
    orphans = orphan_salvage(func_rows, learning_entries)
    refactors = refactor_candidates(func_rows)

    # aggregate metrics
    deeper = {
        'generated': datetime.now(timezone.utc).isoformat(),
        'timeline_buckets': len(timeline),
        'median_quality': sorted(f['quality_score'] for f in func_rows)[len(func_rows)//2] if func_rows else None,
        'top_quality_examples': [r['id'] for r in sorted(func_rows, key=lambda x: -x['quality_score'])[:20]],
        'orphan_salvage_count': len(orphans),
        'refactor_candidate_count': len(refactors),
        'top_modules_by_coupling': centrality[:15],
        'evolution_chain_count': len(maturation),
        'longest_chain': maturation[0]['base'] if maturation else None
    }

    # write artifacts
    with open(ROOT/'deeper_metrics.json','w', encoding='utf-8') as f:
        json.dump(deeper, f, indent=2)
    with open(ROOT/'function_quality.csv','w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','file','name','quality_score','complexity','length','doc_present','cfd_tags','parameters'])
        for r in sorted(func_rows, key=lambda x: -x['quality_score']):
            w.writerow([r['id'], r['file'], r['name'], r['quality_score'], r['complexity'], r['length'], r['doc_present'], '|'.join(r['cfd_tags']), '|'.join(r['parameters'])])
    with open(ROOT/'module_centrality.json','w', encoding='utf-8') as f:
        json.dump(centrality, f, indent=2)
    with open(ROOT/'timeline_metrics.json','w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2)

    # narrative report
    lines = ['# Deeper Analysis Report', f"Generated: {deeper['generated']}", '']
    lines.append('## Temporal Trends')
    if timeline:
        lines.append('Buckets: ' + ', '.join(t['bucket'] for t in timeline[:12]))
        lines.append('Recent bucket failure rates: ' + ', '.join(f"{t['bucket']}={t['failure_rate']*100:.1f}%" for t in timeline[-5:]))
    lines.append('\n## Function Quality')
    lines.append(f"Median quality score: {deeper['median_quality']}")
    lines.append(f"Top 5 functions: {', '.join(r['name'] for r in sorted(func_rows, key=lambda x: -x['quality_score'])[:5])}")
    lines.append(f"Orphan salvage candidates: {len(orphans)} (listed below if any)")
    for o in orphans[:15]:
        lines.append(f"- {o['file']}::{o['name']} score={o['quality_score']}")
    lines.append('\n## Refactor Priorities (Top 10)')
    for r in refactors[:10]:
        why = []
        if r['complexity']>15: why.append('high complexity')
        if r['length']>80: why.append('long')
        if not r['doc_present']: why.append('no doc')
        lines.append(f"- {r['file']}::{r['name']} (priority {r['priority']}) [{' '.join(why)}]")
    lines.append('\n## Module Centrality')
    lines.append('Top modules by coupling: ' + ', '.join(f"{c['module']}({c['coupling']})" for c in centrality[:10]))
    lines.append('\n## Evolution Chains')
    lines.append(f"Total chains: {len(maturation)}")
    for m in maturation[:5]:
        lines.append(f"- {m['base']} length={m['length']} funcΔ={m['func_count_delta']} classΔ={m['class_count_delta']}")
    lines.append('\n## Recommendations')
    lines.append('- Extract shared logic from high-coupling central modules into stable APIs to reduce ripple risk.')
    lines.append('- Refactor top priority functions before adding new features; high complexity correlates with future failures.')
    lines.append('- Promote orphan high-quality functions from failing files into a utilities/core module.')
    lines.append('- Monitor failure_rate trend; reduce by instituting pre-commit complexity guardrails.')
    with open(ROOT/'deeper_analysis_report.md','w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Deeper analysis artifacts generated:')
    for p in ['deeper_metrics.json','function_quality.csv','module_centrality.json','timeline_metrics.json','deeper_analysis_report.md']:
        print(' -', p)

if __name__ == '__main__':
    main()
