"""Extended Learning Dataset Generator

Builds on outputs from `generate_learning_dataset.py` adding deeper CFD / AI workflow analytics:
Outputs:
  - extended_learning_dataset.jsonl : per-Python-file rich function-level metadata
  - dependency_graph.json           : directed module import graph
  - import_cooccurrence.json        : matrix of import co-usage frequencies
  - evolution_chains.json           : inferred prototype->final chains by name/time heuristics
  - failure_taxonomy.json           : clustered failure reason snippets
  - function_catalog.csv            : tabular export of functions (file, name, complexity, size, cfd_tags)
  - pattern_snippets.json           : frequent function signature stems (for scaffold reuse)
  - cfd_scaffold_recommendation.md  : recommended minimal modern CFD project scaffold based on corpus
  - extended_report.md              : human readable deep analysis

Heuristics only (offline). No external API calls.
"""
from __future__ import annotations
import os, re, json, ast, hashlib, csv
from pathlib import Path
from datetime import datetime, timezone
import warnings
from collections import defaultdict, Counter

ROOT = Path(__file__).parent
# Suppress noisy SyntaxWarnings (e.g., invalid escape sequences in docstrings) during AST parsing
warnings.filterwarnings('ignore', category=SyntaxWarning)
PY_FILES: list[Path] = []
CFD_KEYWORDS = ['navier', 'stokes', 'cfd', 'solver', 'flux', 'residual', 'mesh', 'boundary', 'pressure', 'velocity', 'timestep', 'courant']
EXCLUDED_DIR_NAMES = {'.venv','venv','env','site-packages','dist-packages','__pycache__','build','egg-info','node_modules','.git'}

def sha16(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

def list_py() -> None:
    for d,dirnames,files in os.walk(ROOT):
        dirnames[:] = [x for x in dirnames if x not in EXCLUDED_DIR_NAMES]
        for f in files:
            if f.endswith('.py'):
                p = Path(d)/f
                if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
                    continue
                PY_FILES.append(p)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return ''

BRANCH_TOKENS = {'if','for','while','and','or','elif','except','try','with'}

def func_complexity(src_lines: list[str], start: int, end: int) -> int:
    # naive cyclomatic approx = 1 + count of branching tokens in function slice
    slice_text = '\n'.join(src_lines[start-1:end])
    return 1 + sum(slice_text.count(tok + ' ') for tok in BRANCH_TOKENS)

def cfd_tags(name: str, text: str) -> list[str]:
    low = (name + ' ' + text).lower()
    return sorted({kw for kw in CFD_KEYWORDS if kw in low})

SUPPRESSED_SYNTAX_WARNINGS = 0

def extract_functions(p: Path) -> list[dict]:
    code = read_text(p)
    if not code.strip():
        return []
    global SUPPRESSED_SYNTAX_WARNINGS
    try:
        with warnings.catch_warnings(record=True) as w:  # capture (suppressed) syntax warnings per file
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(code, filename=str(p))
            # count only SyntaxWarnings
            SUPPRESSED_SYNTAX_WARNINGS += sum(1 for wi in w if issubclass(wi.category, SyntaxWarning))
    except Exception:
        return []
    lines = code.splitlines()
    funcs: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, 'end_lineno', start)
            body_len = max(0, end-start+1)
            comp = func_complexity(lines, start, end)
            doc = ast.get_docstring(node) or ''
            name = node.name
            text_slice = '\n'.join(lines[start-1:end])
            funcs.append({
                'id': sha16(f"{p}:{name}:{start}"),
                'file': str(p),
                'name': name,
                'lineno': start,
                'end_lineno': end,
                'length': body_len,
                'complexity': comp,
                'doc_present': bool(doc),
                'cfd_tags': cfd_tags(name, text_slice),
                'parameters': [arg.arg for arg in (node.args.args if hasattr(node.args,'args') else [])],
            })
    return funcs

def build_dependency_graph(py_files: list[Path]):
    graph: dict[str, list[str]] = {}
    import_co = Counter()
    for p in py_files:
        code = read_text(p)
        try:
            tree = ast.parse(code, filename=str(p))
        except Exception:
            continue
        mod = module_name(p)
        deps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    deps.append(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split('.')[0]
                    deps.append(top)
        graph[mod] = sorted(set(deps))
        for a in set(deps):
            for b in set(deps):
                if a < b:
                    import_co[(a,b)] += 1
    return graph, import_co

def module_name(p: Path) -> str:
    rel = p.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    return '.'.join(parts)

TIME_PAT = re.compile(r'(\d{6}[_-]?\d{6}|20\d{6})')

def time_hint(name: str) -> float:
    m = TIME_PAT.search(name)
    if m:
        token = m.group(1)
        try:
            if '_' in token or '-' in token and len(token) == 13:
                token = token.replace('-', '_')
                dt = datetime.strptime(token, '%y%m%d_%H%M%S')
                return dt.timestamp()
            if len(token)==8 and token.startswith('20'):
                dt = datetime.strptime(token, '%Y%m%d')
                return dt.timestamp()
        except Exception:
            pass
    return 0.0

def evolution_chains(py_files: list[Path]):
    # Group by stem ignoring version-like suffixes
    groups = defaultdict(list)
    for p in py_files:
        stem = p.stem.lower()
        base = re.sub(r'(v|_)?\d+(?:_?\d+)*$', '', stem)
        groups[base].append(p)
    chains = []
    for base, files in groups.items():
        if len(files) < 2:
            continue
        if base in {'__init__','__main__','setup','test','tests','conftest'}:
            continue
        ordered = sorted(files, key=lambda f: (time_hint(f.name), f.stat().st_mtime))
        chains.append({
            'base': base,
            'sequence': [f.name for f in ordered],
            'count': len(ordered)
        })
    chains.sort(key=lambda c: -c['count'])
    return chains

def cluster_failures():
    # Load existing learning_dataset.jsonl if present to harvest fail reasons
    src = ROOT / 'learning_dataset.jsonl'
    clusters: dict[str, dict] = {}
    if not src.exists():
        return {}
    with src.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if 'fail_reasons' in obj and obj.get('fail_reasons'):
                for fr in obj['fail_reasons']:
                    key = normalize_failure(fr)
                    c = clusters.setdefault(key, {'canonical': fr, 'count':0, 'examples':[]})
                    c['count'] +=1
                    if len(c['examples'])<5:
                        c['examples'].append(fr)
    return clusters

FAIL_SIMPLIFY_PAT = re.compile(r'\d+|0x[0-9a-fA-F]+')

def normalize_failure(text: str) -> str:
    t = text.lower()
    t = FAIL_SIMPLIFY_PAT.sub('#', t)
    tokens = [w for w in re.split(r'[^a-z#]+', t) if w]
    return ' '.join(tokens[:12])

def frequent_signature_stems(funcs: list[dict]):
    stems = Counter()
    for f in funcs:
        params = ','.join(f['parameters'])
        stem = f["name"] + '(' + params + ')'
        # Abstract numeric param names
        stem = re.sub(r'\d+', 'N', stem)
        stems[stem] +=1
    top = stems.most_common(50)
    return [{'signature': s, 'count': c} for s,c in top]

def build_scaffold_recommendation(funcs: list[dict], deps: dict[str, list[str]]):
    # Heuristic: identify key CFD tag coverage & propose modules
    tag_freq = Counter()
    for f in funcs:
        for t in f['cfd_tags']:
            tag_freq[t]+=1
    core_tags = [t for t,_ in tag_freq.most_common(8)]
    lines = ['# Recommended CFD Scaffold', '', 'Core functional areas detected: ' + ', '.join(core_tags), '', 'Suggested module layout:', '']
    layout = [
        'cfd_core/__init__.py',
        'cfd_core/mesh.py  # mesh generation & boundary tagging',
        'cfd_core/physics.py  # flux computations, material properties',
        'cfd_core/solver.py  # time-stepping orchestration',
        'cfd_core/boundary_conditions.py',
        'cfd_core/postprocess.py',
        'experiments/    # prototype and research scripts',
        'scripts/run_case.py',
        'tests/test_solver.py',
        'tests/test_fluxes.py'
    ]
    lines += [f'- {p}' for p in layout]
    lines.append('')
    lines.append('Implementation guidance:')
    lines.append('- Keep functions < 60 lines; refactor when complexity > 12.')
    lines.append('- Centralize external imports; aim to reduce rarely reused ones.')
    lines.append('- Record experiment metadata (params, hash) to reproducibility log.')
    return '\n'.join(lines)

def main():
    list_py()
    all_funcs: list[dict] = []
    for p in PY_FILES:
        all_funcs.extend(extract_functions(p))
    dep_graph, import_co = build_dependency_graph(PY_FILES)
    chains = evolution_chains(PY_FILES)
    fail_clusters = cluster_failures()
    sig_stems = frequent_signature_stems(all_funcs)
    scaffold_md = build_scaffold_recommendation(all_funcs, dep_graph)

    # Write artifacts
    with open(ROOT/'extended_learning_dataset.jsonl','w', encoding='utf-8') as f:
        for func in all_funcs:
            f.write(json.dumps(func)+'\n')
    with open(ROOT/'dependency_graph.json','w', encoding='utf-8') as f:
        json.dump(dep_graph, f, indent=2)
    with open(ROOT/'import_cooccurrence.json','w', encoding='utf-8') as f:
        json.dump({f"{a}|{b}":c for (a,b),c in import_co.items()}, f, indent=2)
    with open(ROOT/'evolution_chains.json','w', encoding='utf-8') as f:
        json.dump(chains, f, indent=2)
    with open(ROOT/'failure_taxonomy.json','w', encoding='utf-8') as f:
        json.dump(fail_clusters, f, indent=2)
    with open(ROOT/'pattern_snippets.json','w', encoding='utf-8') as f:
        json.dump(sig_stems, f, indent=2)
    with open(ROOT/'function_catalog.csv','w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id','file','name','lineno','end_lineno','length','complexity','doc_present','cfd_tags','parameters'])
        for func in all_funcs:
            w.writerow([func['id'], func['file'], func['name'], func['lineno'], func['end_lineno'], func['length'], func['complexity'], func['doc_present'], '|'.join(func['cfd_tags']), '|'.join(func['parameters'])])
    with open(ROOT/'cfd_scaffold_recommendation.md','w', encoding='utf-8') as f:
        f.write(scaffold_md)

    # Extended report
    report_lines = ['# Extended Learning Report', 'Generated: '+datetime.now(timezone.utc).isoformat(),'']
    report_lines.append(f"Python files analyzed: {len(PY_FILES)}")
    report_lines.append(f"Suppressed SyntaxWarnings during parsing: {SUPPRESSED_SYNTAX_WARNINGS}")
    report_lines.append(f"Total functions extracted: {len(all_funcs)}")
    # Complexity dist
    comp_bins = Counter()
    for func in all_funcs:
        c = func['complexity']
        bucket = '<=5' if c<=5 else '<=10' if c<=10 else '<=15' if c<=15 else '<=25' if c<=25 else '>25'
        comp_bins[bucket]+=1
    report_lines.append('Complexity distribution: '+', '.join(f"{k}:{v}" for k,v in sorted(comp_bins.items())))
    # CFD tag freq
    tag_freq = Counter(t for f in all_funcs for t in f['cfd_tags'])
    report_lines.append('Top CFD tags: '+', '.join(f"{t}({c})" for t,c in tag_freq.most_common(15)))
    report_lines.append('Top signature stems: '+', '.join(f"{s['signature']}({s['count']})" for s in sig_stems[:10]))
    report_lines.append('Evolution chains (top 5 by length):')
    for ch in chains[:5]:
        report_lines.append('- '+ch['base']+': '+' -> '.join(ch['sequence']))
    report_lines.append('Failure clusters (top 10):')
    top_fail = sorted(fail_clusters.items(), key=lambda x: -x[1]['count'])[:10]
    for k,(key,data) in enumerate(top_fail):  # type: ignore
        report_lines.append(f"- {data['canonical']} (count {data['count']})")
    report_lines.append('\nSee cfd_scaffold_recommendation.md for scaffold guidance.')
    with open(ROOT/'extended_report.md','w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('Extended learning artifacts generated:')
    for p in ['extended_learning_dataset.jsonl','dependency_graph.json','import_cooccurrence.json','evolution_chains.json','failure_taxonomy.json','function_catalog.csv','pattern_snippets.json','cfd_scaffold_recommendation.md','extended_report.md']:
        print(' -', p)

if __name__ == '__main__':
    main()
