"""Build a lightweight TF-IDF semantic index over functions.jsonl for improved retrieval.
Artifacts:
  knowledge_db/tfidf_index.json  (vocab, idf list, function_vectors: list of (id, sparse_pairs))
"""
from __future__ import annotations
import json, math, re
from pathlib import Path

ROOT = Path(__file__).parent
KB = ROOT/'knowledge_db'
FUNCS = KB/'functions.jsonl'
OUT = KB/'tfidf_index.json'
TOP_VOCAB = 5000

TOKEN_RE = re.compile(r'[a-zA-Z_]{3,}')
STOP = set(['the','and','for','from','with','this','that','init','self','none'])

def load_funcs():
    out=[]
    if not FUNCS.exists():
        return out
    for line in FUNCS.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except: # noqa
            pass
    return out

def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP]

def main():
    funcs = load_funcs()
    if not funcs:
        print('No functions.jsonl; run build_cfd_knowledge_db.py first')
        return
    # doc tokens
    docs = []
    df = {}
    for f in funcs:
        text = f['name'] + ' ' + ' '.join(f.get('cfd_tags',[])) + ' ' + (f.get('snippet','')[:800])
        toks = tokenize(text)
        uniq = set(toks)
        for u in uniq:
            df[u] = df.get(u,0)+1
        docs.append((f['id'], toks))
    # vocab selection
    vocab_terms = sorted(df.items(), key=lambda x: -x[1])[:TOP_VOCAB]
    vocab = {term:i for i,(term,_) in enumerate(vocab_terms)}
    N = len(docs)
    idf = [0.0]*len(vocab)
    for term,i in vocab.items():
        idf[i] = math.log((1+N)/(1+df[term])) + 1.0
    vectors = []
    for fid, toks in docs:
        tf = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t,0)+1
        if not tf:
            continue
        # build sparse pairs (index, weight)
        total = sum(tf.values())
        pairs=[]
        for t, c in tf.items():
            idx = vocab[t]
            w = (c/total) * idf[idx]
            pairs.append([idx, round(w,6)])
        vectors.append({'id': fid, 'v': pairs})
    OUT.write_text(json.dumps({'vocab':[t for t,_ in vocab_terms], 'idf': idf, 'vectors': vectors}, indent=2), encoding='utf-8')
    print('Semantic index built:', OUT)

if __name__ == '__main__':
    main()
