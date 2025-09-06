"""Playground: Multi-Objective Tuner Demo (v0.1)
Generates synthetic parameter configurations and evaluates Pareto frontier for objectives:
- convergence_time (minimize)
- final_residual (minimize)
- stability_failures (minimize)
Demonstrates Pareto filtering & dominance count.
"""
from __future__ import annotations
import argparse, json, os, random

SCHEMA_VERSION=1

# Synthetic evaluation: treat parameters relax_u, relax_p, dt_safety in [0,1]
# Surfaces are contrived: lower residual near mid region, time faster at higher relax but risk of failures.

def evaluate(param):
    relax_u, relax_p, dt_safety = param
    convergence_time = 1000 * (1.2 - 0.5*(relax_u+relax_p)/2) * (1.0 + 0.3*(1-dt_safety))
    final_residual = 1e-3 * (1 + abs(relax_u-0.6) + abs(relax_p-0.6))* (1+0.5*(0.7-dt_safety))
    risk = max(0, (relax_u+relax_p)/2 - 0.85) + max(0, dt_safety - 0.8)
    stability_failures = 1 if random.random() < 0.5*risk else 0
    return convergence_time, final_residual, stability_failures

def dominates(a,b):
    return all(x<=y for x,y in zip(a['obj'], b['obj'])) and any(x<y for x,y in zip(a['obj'], b['obj']))

def pareto(configs):
    front=[]
    for c in configs:
        if any(dominates(o,c) for o in configs if o is not c):
            continue
        front.append(c)
    return front

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-dir', default='playground/artifacts/multi_objective')
    args=ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    configs=[]
    for _ in range(args.samples):
        param=(random.random(), random.random(), random.random())
        obj=evaluate(param)
        configs.append({'param':param,'obj':obj})
    front=pareto(configs)
    # dominance count
    for c in configs:
        c['dominated_by']=sum(1 for o in configs if dominates(o,c) and o is not c)
    out={'schema_version':SCHEMA_VERSION,'total':len(configs),'pareto_count':len(front),'pareto':front[:25]}
    with open(os.path.join(args.output_dir,'multi_objective_frontier.json'),'w') as f:
        json.dump(out,f,indent=2)
    print(json.dumps({'pareto_count':len(front),'total':len(configs)}))

if __name__=='__main__':
    main()
