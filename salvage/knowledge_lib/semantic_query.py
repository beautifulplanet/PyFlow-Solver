"""Query the TF-IDF semantic index for similar functions.
Usage:
  python semantic_query.py "pressure correction matrix assembly"
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
IDX = ROOT/'knowledge_db'/'tfidf_index.json'
FUNCS = ROOT/'knowledge_db'/'functions.jsonl'
TOKEN_RE = re.compile(r'[a-zA-Z_]{3,}')

def load_index():
    if not IDX.exists():
        print('Index missing; run build_semantic_index.py')
        sys.exit(1)
    return json.loads(IDX.read_text(encoding='utf-8'))

def load_func_meta():
    meta={}
    if not FUNCS.exists():
        return meta
    for line in FUNCS.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            meta[obj['id']] = obj
        except: # noqa
            pass
    return meta

def tokenize(q: str):
    return [t.lower() for t in TOKEN_RE.findall(q)]

def build_query_vector(q_tokens, vocab, idf):
    counts={}
    for t in q_tokens:
        if t in vocab:
            counts[t] = counts.get(t,0)+1
    total = sum(counts.values()) or 1
    pairs=[]
    for t,c in counts.items():
        idx = vocab[t]
        w = (c/total) * idf[idx]
        pairs.append((idx,w))
    norm = math.sqrt(sum(w*w for _,w in pairs)) or 1
    return {i: w/norm for i,w in pairs}

def vector_norm(vec_pairs):
    return math.sqrt(sum(w*w for _,w in vec_pairs)) or 1

def main():
    if len(sys.argv)<2:
        print(__doc__)
        return
    query = ' '.join(sys.argv[1:])
    data = load_index()
    meta = load_func_meta()
    vocab = {t:i for i,t in enumerate(data['vocab'])}
    idf = data['idf']
    q_vec = build_query_vector(tokenize(query), vocab, idf)
    q_norm = math.sqrt(sum(v*v for v in q_vec.values())) or 1
    scores=[]
    for item in data['vectors']:
        dot=0.0
        for idx,w in item['v']:
            if idx in q_vec:
                dot += w * q_vec[idx]
        denom = (math.sqrt(sum(w*w for _,w in item['v'])) or 1) * q_norm
        sim = dot/denom if denom else 0.0
        scores.append((sim, item['id']))
    scores.sort(key=lambda x: -x[0])
    for sim, fid in scores[:10]:
        f = meta.get(fid, {})
        print(f"{sim:.3f} | {f.get('name')}({', '.join(f.get('parameters',[]))}) -> {f.get('file')}")
    
if __name__ == '__main__':
    main()
