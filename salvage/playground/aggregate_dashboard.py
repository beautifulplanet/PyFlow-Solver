"""Playground: Aggregate Dashboard (v0.1)
Aggregates artifacts from playground scripts and emits a compact summary JSON.
"""
from __future__ import annotations
import argparse, json, os, glob

SCHEMA_VERSION=1

PATTERNS={
 'manufactured': 'playground/artifacts/manufactured/manufactured_results.json',
 'adaptive_dt': 'playground/artifacts/adaptive_dt/adaptive_dt_pid_metrics.json',
 'plateau': 'playground/artifacts/plateau/plateau_classifier_metrics.json',
 'stencil': 'playground/artifacts/stencil/stencil_audit.json',
 'duplication': 'playground/artifacts/duplication/duplication_similarity.json',
 'microbench': 'playground/artifacts/microbench/microbench_results.json',
 'embedding': 'playground/artifacts/embedding/embedding_similarity.json',
 'failure_injection': 'playground/artifacts/failure_injection/failure_injection_metrics.json',
 'multi_objective': 'playground/artifacts/multi_objective/multi_objective_frontier.json'
}

def load(path):
    try:
        with open(path,'r') as f:
            return json.load(f)
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', default='playground/artifacts/dashboard_summary.json')
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary={'schema_version':SCHEMA_VERSION}
    for k, p in PATTERNS.items():
        obj=load(p)
        summary[k]=obj if obj is not None else {'status':'missing'}
    with open(args.output,'w') as f:
        json.dump(summary,f,indent=2)
    print(json.dumps({'status':'ok','keys':list(summary.keys())}))

if __name__=='__main__':
    main()
