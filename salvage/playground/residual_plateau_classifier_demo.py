"""Playground: Residual Plateau Classifier Demo (v0.1)
Generates synthetic residual sequences (monotonic, oscillatory, plateau, divergent) and evaluates classifier precision/recall.
"""
from __future__ import annotations
import argparse, json, math, os, random, statistics

SCHEMA_VERSION = 1

MODES = ['monotonic','oscillatory','plateau','divergent']

def synthesize(mode, steps, noise, seed):
    random.seed(seed)
    r=[]
    val=1.0
    for k in range(steps):
        if mode=='monotonic':
            val *= 0.99
        elif mode=='oscillatory':
            val *= 0.995 + 0.005*math.sin(k/5)
        elif mode=='plateau':
            if k< steps//3:
                val *= 0.96
            else:
                val *= 0.9995
        elif mode=='divergent':
            val *= 1.005
        val += random.gauss(0, noise)
        val = max(val, 1e-14)
        r.append(val)
    return r

def classify(sequence, window=40, slope_thresh=-0.1, plateau_slope=-0.02):
    import math
    if len(sequence)<window+2:
        return 'unknown'
    import numpy as np
    import numpy.linalg as la
    y = [math.log(max(v,1e-30)) for v in sequence[-window:]]
    x = list(range(window))
    xbar = sum(x)/window; ybar = sum(y)/window
    num = sum((xi - xbar)*(yi - ybar) for xi,yi in zip(x,y))
    den = sum((xi - xbar)**2 for xi in x)
    slope = num/den if den else 0
    # divergence check
    if sequence[-1] > sequence[0]*10:
        return 'divergent'
    if slope > slope_thresh:
        return 'divergent'
    if slope > plateau_slope:
        return 'plateau'
    # oscillation heuristic
    # compute zero-crossings of first difference around smoothed trend
    diffs = [sequence[i+1]-sequence[i] for i in range(len(sequence)-1)]
    zero_cross = sum(1 for i in range(len(diffs)-1) if diffs[i]==0 or diffs[i]*diffs[i+1]<0)
    if zero_cross > window/6:
        return 'oscillatory'
    return 'monotonic'

def evaluate(runs):
    tp=0; total=0; confusion={}
    for rec in runs:
        total+=1
        key=(rec['mode'], rec['pred'])
        confusion[key]=confusion.get(key,0)+1
        if rec['mode']==rec['pred']:
            tp+=1
    acc=tp/total if total else 0
    return acc, confusion

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--noise', type=float, default=1e-4)
    ap.add_argument('--output-dir', default='playground/artifacts/plateau')
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runs=[]
    for mode in MODES:
        for s in range(args.seeds):
            seq=synthesize(mode, args.steps, args.noise, seed= (hash(mode)+s) & 0xffffffff)
            pred=classify(seq)
            runs.append({'mode':mode,'pred':pred})
    acc,conf=evaluate(runs)
    summary={'schema_version':SCHEMA_VERSION,'accuracy':acc,'confusion':conf,'total':len(runs)}
    with open(os.path.join(args.output_dir,'plateau_classifier_metrics.json'),'w') as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary))

if __name__=='__main__':
    main()
