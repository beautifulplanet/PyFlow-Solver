"""Playground: Adaptive dt PID Controller Simulation (v0.1)
Simulates convergence residual evolution under a model acknowledging dt influence, testing PID gains effects.
"""
from __future__ import annotations
import argparse, json, math, os, random
import numpy as np

SCHEMA_VERSION = 1

# Simple synthetic residual model: r_{k+1} = r_k * exp(-alpha * dt_eff) + noise
# dt_eff = dt * stability_factor; if dt too large, stability_factor drops.

def simulate(steps, dt0, target_decay, pid, noise_level, seed=0):
    random.seed(seed); np.random.seed(seed)
    Kp, Ki, Kd = pid
    residuals = []
    dt = dt0
    integ = 0.0
    prev_err = None
    r = 1.0
    for k in range(steps):
        # model stability factor
        stability = 1.0 if dt < 0.01 else max(0.1, 0.02/dt)
        r = r * math.exp(-0.5 * stability * dt) + random.gauss(0, noise_level)
        r = max(r, 1e-14)
        # error vs target decay per step (ideal geometric)
        ideal = math.exp(-target_decay * k)
        err = r - ideal
        integ += err
        deriv = 0 if prev_err is None else (err - prev_err)
        prev_err = err
        adj = Kp*err + Ki*integ + Kd*deriv
        # adjust dt bounded
        dt = dt * math.exp(-adj)
        dt = min(max(dt, 1e-5), 0.05)
        residuals.append({'step': k, 'r': r, 'dt': dt, 'err': err})
    return residuals

def evaluate(residuals):
    import statistics
    rs = [r['r'] for r in residuals]
    dts = [r['dt'] for r in residuals]
    final = rs[-1]
    var_dt = statistics.pvariance(dts)
    mean_dt = statistics.mean(dts)
    return {
        'final_residual': final,
        'dt_coeff_var': math.sqrt(var_dt)/mean_dt if mean_dt else 0.0,
        'steps': len(residuals)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=float, nargs=3, default=[5.0, 0.0, 0.1])
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--dt0', type=float, default=0.005)
    ap.add_argument('--target-decay', type=float, default=0.01)
    ap.add_argument('--noise', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-dir', default='playground/artifacts/adaptive_dt')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    residuals = simulate(args.steps, args.dt0, args.target_decay, args.pid, args.noise, args.seed)
    metrics = evaluate(residuals)
    out = {
        'schema_version': SCHEMA_VERSION,
        'pid': args.pid,
        'metrics': metrics,
        'sample': residuals[0:10]
    }
    with open(os.path.join(args.output_dir, 'adaptive_dt_pid_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    # stream
    for rec in residuals:
        print(json.dumps(rec))

if __name__ == '__main__':
    main()
