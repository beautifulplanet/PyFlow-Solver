"""Playground: Duplication Similarity Demo (v0.1)
Computes TF-IDF cosine similarity between synthetic or user-provided snippets to illustrate threshold tuning.
"""
from __future__ import annotations
import argparse, json, os, math, re
from collections import Counter
import itertools

SCHEMA_VERSION=1
TOKEN_RE=re.compile(r'[A-Za-z_]{2,}')

stop = set(['the','and','for','def','return','import','from','as','if','in','None','True','False'])

def tokenize(text:str):
    return [t.lower() for t in TOKEN_RE.findall(text) if t not in stop]

def build_corpus(snippets):
    docs=[tokenize(s) for s in snippets]
    df=Counter()
    for doc in docs:
        for term in set(doc):
            df[term]+=1
    N=len(docs)
    vectors=[]
    for doc in docs:
        tf=Counter(doc)
        vec={}
        for term,count in tf.items():
            idf=math.log((N+1)/(df[term]+1))+1
            vec[term]= (count/len(doc))*idf
        vectors.append(vec)
    return vectors, df

def cosine(a,b):
    keys=set(a.keys())|set(b.keys())
    dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
    na=math.sqrt(sum(v*v for v in a.values()))
    nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snippets', nargs='*', help='Inline snippets or file paths (detected by existing file)')
    ap.add_argument('--threshold-warn', type=float, default=0.85)
    ap.add_argument('--threshold-block', type=float, default=0.90)
    ap.add_argument('--output-dir', default='playground/artifacts/duplication')
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw=[]
    if not args.snippets:
        # Provide default synthetic set
        raw=[
            'def compute_grad(u): return (u[2:]-u[:-2])*0.5',
            'def compute_gradient(u): return 0.5*(u[2:]-u[:-2])',
            'def apply_bc(u): u[0]=u[1]; u[-1]=u[-2]; return u',
            'def unrelated(a,b): return a*b + 2',
        ]
    else:
        for s in args.snippets:
            if os.path.isfile(s):
                with open(s,'r',errors='ignore') as f:
                    raw.append(f.read())
            else:
                raw.append(s)
    vectors,_=build_corpus(raw)
    pairs=[]
    for (i,va),(j,vb) in itertools.combinations(list(enumerate(vectors)),2):
        sim=cosine(va,vb)
        status='ok'
        if sim>=args.threshold_block:
            status='block'
        elif sim>=args.threshold_warn:
            status='warn'
        pairs.append({'i':i,'j':j,'similarity':sim,'status':status})
    out={'schema_version':SCHEMA_VERSION,'pairs':pairs,'threshold_warn':args.threshold_warn,'threshold_block':args.threshold_block}
    with open(os.path.join(args.output_dir,'duplication_similarity.json'),'w') as f:
        json.dump(out,f,indent=2)
    for p in pairs:
        print(json.dumps(p))

if __name__=='__main__':
    main()
