"""Build a lightweight on-disk CFD knowledge base for assistant queries.

Steps:
 1. Load orphan_salvage.jsonl (salvaged high-quality functions) and learning_dataset.jsonl.
 2. Extract code snippets for top N salvaged items and consolidate into cfd_core/reusable_funcs.py.
 3. Generate refactor_plan.md grouping oversize functions by theme (solver, assembly, pressure, generic).
 4. Build knowledge_db/ index JSON files:
      - functions.jsonl (id, name, file, quality, tags, evolution_tag, snippet)
      - files.jsonl (path, labels, evolution_tag, keyword_hits, fail_reasons[:3])
      - vocab.json (top keyword frequencies, tag frequencies)
 5. Provide a simple QA script skeleton (qa_query.py) that does naive retrieval by keyword/tag overlap.

Outputs under ./knowledge_db
"""
from __future__ import annotations
import json, os, re, ast
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent
print('[build_cfd_knowledge_db] ROOT =', ROOT)
KB_DIR = ROOT/'knowledge_db'
if not KB_DIR.exists():
    print('[build_cfd_knowledge_db] os.makedirs for KB_DIR', KB_DIR)
    os.makedirs(KB_DIR, exist_ok=True)
SALVAGE = ROOT/'orphan_salvage.jsonl'
LEARN = ROOT/'learning_dataset.jsonl'
REUSABLE_DIR = ROOT/'cfd_core'
if not REUSABLE_DIR.exists():
    print('[build_cfd_knowledge_db] os.makedirs for reusable dir', REUSABLE_DIR)
    os.makedirs(REUSABLE_DIR, exist_ok=True)
REUSABLE_FILE = REUSABLE_DIR/'reusable_funcs.py'

TOP_N_FUNCS = 300

def load_jsonl(path: Path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip(): continue
        try: out.append(json.loads(line))
        except Exception: pass
    return out

def read_file(path: Path):
    try: return path.read_text(encoding='utf-8', errors='replace')
    except Exception: return ''

def extract_function_source(file_path: str, name: str, lineno: int, end_lineno: int) -> str:
    p = Path(file_path)
    txt = read_file(p)
    if not txt: return ''
    lines = txt.splitlines()
    # safe slice indexes
    start = max(1, lineno)-1
    end = min(len(lines), end_lineno)
    snippet = '\n'.join(lines[start:end])
    return snippet

def theme_for(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ['solve','solver','step']): return 'solver'
    if 'assemble' in low or 'matrix' in low: return 'assembly'
    if 'pressure' in low: return 'pressure'
    if 'velocity' in low or 'residual' in low: return 'field'
    return 'generic'

def build_reusable_module(funcs):
    header = """Reusable CFD Functions (auto-generated)

DO NOT EDIT BY HAND. Source: orphan_salvage.jsonl top quality subset.
"""
    lines = [header, 'from __future__ import annotations', '']
    for f in funcs:
        snippet = extract_function_source(f['file'], f['name'], int(f['lineno']), int(f['end_lineno']))
        if not snippet.strip():
            continue
        # ensure snippet starts with def
        if not snippet.lstrip().startswith('def '):
            continue
        lines.append(f'# source: {f["file"]}:{f["lineno"]}-{f["end_lineno"]} quality={f.get("quality_score")}' )
        lines.append(snippet)
        lines.append('')
    REUSABLE_FILE.write_text('\n'.join(lines), encoding='utf-8')

def build_refactor_plan():
    report = Path(ROOT/'PRECOMMIT_COMPLEXITY_REPORT.md')
    if not report.exists(): return
    lines = report.read_text(encoding='utf-8', errors='ignore').splitlines()
    buckets = defaultdict(list)
    for line in lines:
        if not line.startswith('- '): continue
        # - path::func complexity=... length=...
        m = re.match(r'- (.+?)::([\w_]+) complexity=(\d+) length=(\d+)', line)
        if not m: continue
        path, name, comp, length = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        theme = theme_for(name)
        buckets[theme].append({'path':path,'name':name,'complexity':comp,'length':length})
    out = ['# Refactor Plan','Grouped by theme (solver, assembly, pressure, field, generic).','']
    for theme, items in buckets.items():
        out.append(f'## {theme} ({len(items)})')
        items.sort(key=lambda x: (-x['complexity'], -x['length']))
        for it in items[:80]:
            out.append(f"- {it['path']}::{it['name']} complexity={it['complexity']} length={it['length']}")
        out.append('')
    (ROOT/'refactor_plan.md').write_text('\n'.join(out), encoding='utf-8')

def build_knowledge_db():
    salvage = load_jsonl(SALVAGE)
    learn = load_jsonl(LEARN)
    # index files metadata
    with (KB_DIR/'files.jsonl').open('w', encoding='utf-8') as f:
        for e in learn:
            f.write(json.dumps({
                'path': e.get('path'),
                'labels': e.get('labels',[]),
                'evolution_tag': e.get('evolution_tag'),
                'keyword_hits': e.get('keyword_hits',[])[:12],
                'fail_reasons': e.get('fail_reasons',[])[:3]
            })+'\n')
    # function entries (take top N)
    salvage.sort(key=lambda x: -x.get('quality_score',0))
    top_funcs = salvage[:TOP_N_FUNCS]
    # gather vocab
    tag_freq = Counter()
    name_tokens = Counter()
    with (KB_DIR/'functions.jsonl').open('w', encoding='utf-8') as f:
        for func in top_funcs:
            snippet = extract_function_source(func['file'], func['name'], int(func['lineno']), int(func['end_lineno']))
            tag_freq.update(func.get('cfd_tags',[]))
            name_tokens.update(re.findall(r'[a-zA-Z_]{4,}', func['name']))
            f.write(json.dumps({
                'id': func.get('id'),
                'name': func.get('name'),
                'file': func.get('file'),
                'quality': func.get('quality_score'),
                'cfd_tags': func.get('cfd_tags',[]),
                'parameters': func.get('parameters',[]),
                'complexity': func.get('complexity'),
                'length': func.get('length'),
                'snippet': snippet
            })+'\n')
    vocab = {
        'tag_frequency': tag_freq.most_common(100),
        'name_token_frequency': name_tokens.most_common(200)
    }
    (KB_DIR/'vocab.json').write_text(json.dumps(vocab, indent=2), encoding='utf-8')

def write_qa_script():
    qa = ROOT/'qa_query.py'
    if qa.exists(): return
    qa.write_text('''"""Naive QA retrieval over local CFD knowledge base."
from __future__ import annotations
import json, re, sys, math
from pathlib import Path

ROOT = Path(__file__).parent
KB = ROOT/'knowledge_db'

def load_jsonl(p: Path):
    out=[]
    if not p.exists(): return out
    for line in p.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except: pass
    return out

def score(query_tokens, item_tokens):
    if not item_tokens: return 0.0
    overlap = len(query_tokens & item_tokens)
    return overlap / math.sqrt(len(item_tokens)+1)

def main():
    if len(sys.argv)<2:
        print('Usage: python qa_query.py "your question about CFD"')
        return
    q=' '.join(sys.argv[1:])
    tokens = {t.lower() for t in re.findall(r'[a-zA-Z_]{4,}', q)}
    funcs = load_jsonl(KB/'functions.jsonl')
    scored=[]
    for f in funcs:
        ftokens = {t.lower() for t in re.findall(r'[a-zA-Z_]{4,}', f['name']+' '+' '.join(f.get('cfd_tags',[])))}
        scored.append((score(tokens, ftokens), f))
    scored.sort(key=lambda x: -x[0])
    for s,f in scored[:10]:
        print(f"{s:.2f} | {f['name']}({', '.join(f.get('parameters',[]))}) -> {f['file']}")
        print(f['snippet'].split('\n')[0][:120])
        print('-')

if __name__=='__main__':
    main()
''', encoding='utf-8')

def main():
    salvage = load_jsonl(SALVAGE)
    # Build reusable module from top salvage
    build_reusable_module(salvage[:TOP_N_FUNCS])
    build_refactor_plan()
    build_knowledge_db()
    write_qa_script()
    print('Knowledge DB built:')
    for p in ['cfd_core/reusable_funcs.py','refactor_plan.md','knowledge_db/functions.jsonl','knowledge_db/files.jsonl','knowledge_db/vocab.json','qa_query.py']:
        print(' -', p)

if __name__ == '__main__':
    main()
