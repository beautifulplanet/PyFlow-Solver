"""Playground: Failure Injection Harness (v0.1)
Simulates and classifies injected failure modes using simplified patterns.
"""
from __future__ import annotations
import argparse, json, os, random, math

SCHEMA_VERSION=1
FAILURES=['divergence','plateau','nan','mass_leak','perf_regression','oscillation']

def generate_sequence(failure, steps, seed):
    random.seed(seed)
    r=[]; val=1.0
    for k in range(steps):
        if failure=='divergence':
            val *= 1.01
        elif failure=='plateau':
            if k<steps//4: val*=0.96
            else: val*=0.999
        elif failure=='nan':
            if k==steps//2: return [float('nan')]*(steps-k)
            val*=0.98
        elif failure=='mass_leak':
            val*=0.985
        elif failure=='perf_regression':
            val*=0.97
        elif failure=='oscillation':
            val *= 0.995 + 0.01*math.sin(k/3)
        val += random.gauss(0,1e-4)
        r.append(val)
    return r

def classify(seq):
    import math
    if any(math.isnan(x) for x in seq): return 'nan'
    if seq[-1] > seq[0]*10: return 'divergence'
    # slope
    import numpy as np
    y=[math.log(max(v,1e-30)) for v in seq[-60:]] if len(seq)>60 else [math.log(max(v,1e-30)) for v in seq]
    x=list(range(len(y)))
    xbar=sum(x)/len(x); ybar=sum(y)/len(y)
    num=sum((xi-xbar)*(yi-ybar) for xi,yi in zip(x,y))
    den=sum((xi-xbar)**2 for xi in x)
    slope=num/den if den else 0
    if slope > -0.05: # near flat or growing
        if slope > 0: return 'divergence'
        return 'plateau'
    # oscillation detection
    diffs=[seq[i+1]-seq[i] for i in range(len(seq)-1)]
    zero_cross=sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1]<0)
    if zero_cross > len(diffs)/8: return 'oscillation'
    return 'normal'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--output-dir', default='playground/artifacts/failure_injection')
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results=[]
    for f in FAILURES:
        for r in range(args.runs):
            seq=generate_sequence(f, args.steps, seed=r*1337+hash(f)%10000)
            pred=classify(seq)
            results.append({'failure':f,'pred':pred})
    accuracy=sum(1 for x in results if x['failure']==x['pred'])/len(results)
    confusion={}
    for rec in results:
        key=(rec['failure'],rec['pred'])
        confusion[key]=confusion.get(key,0)+1
    out={'schema_version':SCHEMA_VERSION,'accuracy':accuracy,'confusion':confusion,'total':len(results)}
    with open(os.path.join(args.output_dir,'failure_injection_metrics.json'),'w') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out))

if __name__=='__main__':
    main()
