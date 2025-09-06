"""Naive QA retrieval over local CFD knowledge base."
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
        print(f['snippet'].split('
')[0][:120])
        print('-')

if __name__=='__main__':
    main()
