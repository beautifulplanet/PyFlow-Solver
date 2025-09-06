"""Playground: Embedding Similarity Demo (v0.1)
Attempts to compute sentence embeddings for code snippets; falls back to TF-IDF if no embedding backend is available.

This is illustrative; in production you'd integrate a proper model (e.g., sentence-transformers) behind a feature flag.
"""
from __future__ import annotations
import argparse, json, os, math, re, itertools

SCHEMA_VERSION=1
TOKEN_RE=re.compile(r'[A-Za-z_]{2,}')

def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]

try:
    # Placeholder: try to import a hypothetical embedding provider
    from sentence_transformers import SentenceTransformer  # type: ignore
    _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # type: ignore
except Exception:  # pragma: no cover
    _EMBED_MODEL = None

# Basic TF-IDF fallback
from collections import Counter

def tfidf_vectors(snippets):
    docs=[tokenize(s) for s in snippets]
    df=Counter()
    for d in docs:
        for t in set(d):
            df[t]+=1
    N=len(docs)
    vecs=[]
    for d in docs:
        tf=Counter(d)
        v={}
        for term,c in tf.items():
            idf=math.log((N+1)/(df[term]+1))+1
            v[term]=(c/len(d))*idf
        vecs.append(v)
    return vecs

def cosine_dict(a,b):
    keys=set(a.keys())|set(b.keys())
    dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

def cosine_vec(a,b):  # list or ndarray
    import numpy as np
    a,b = (a/ (np.linalg.norm(a)+1e-12)), (b/(np.linalg.norm(b)+1e-12))
    return float((a*b).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snippets', nargs='*')
    ap.add_argument('--output-dir', default='playground/artifacts/embedding')
    ap.add_argument('--warn', type=float, default=0.80)
    ap.add_argument('--block', type=float, default=0.90)
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.snippets:
        snippets=[
            'def apply_bc(u):\n    u[0]=u[1]; u[-1]=u[-2]; return u',
            'def boundary_apply(u):\n    u[0]=u[1]; u[len(u)-1]=u[-2]; return u',
            'def unrelated(x):\n    return x*x + 3',
        ]
    else:
        # treat inputs as raw text or file paths
        snippets=[]
        for s in args.snippets:
            if os.path.isfile(s):
                with open(s,'r',errors='ignore') as f:
                    snippets.append(f.read())
            else:
                snippets.append(s)
    pairs=[]
    if _EMBED_MODEL is not None:
        try:
            vecs = _EMBED_MODEL.encode(snippets)  # type: ignore
            mode='embedding'
            for (i,a),(j,b) in itertools.combinations(list(enumerate(vecs)),2):
                sim=cosine_vec(a,b)
                status='ok'
                if sim>=args.block: status='block'
                elif sim>=args.warn: status='warn'
                pairs.append({'i':i,'j':j,'similarity':sim,'status':status})
        except Exception:
            _EMBED_MODEL is None  # fallback
    if _EMBED_MODEL is None:
        # fallback TF-IDF
        vecs = tfidf_vectors(snippets)
        mode='tfidf_fallback'
        for (i,a),(j,b) in itertools.combinations(list(enumerate(vecs)),2):
            sim=cosine_dict(a,b)
            status='ok'
            if sim>=args.block: status='block'
            elif sim>=args.warn: status='warn'
            pairs.append({'i':i,'j':j,'similarity':sim,'status':status})
    out={'schema_version':SCHEMA_VERSION,'mode':mode,'pairs':pairs,'warn':args.warn,'block':args.block}
    with open(os.path.join(args.output_dir,'embedding_similarity.json'),'w') as f:
        json.dump(out,f,indent=2)
    for p in pairs:
        print(json.dumps(p))

if __name__=='__main__':
    main()
