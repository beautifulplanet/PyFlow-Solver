"""Playground: Dashboard Markdown Renderer (v0.1)
Reads playground/artifacts/dashboard_summary.json and renders a concise Markdown report.
"""
from __future__ import annotations
import json, os, math

IN_PATH = 'playground/artifacts/dashboard_summary.json'
OUT_PATH = 'playground/artifacts/dashboard_summary.md'

def _h1(s):
    return f"# {s}\n\n"

def _h2(s):
    return f"## {s}\n\n"

def _kv(k,v):
    return f"- {k}: {v}\n"

def _missing():
    return "- status: missing\n"

def render(summary: dict) -> str:
    parts = []
    parts.append(_h1('Playground Dashboard Summary'))
    parts.append('- schema_version: ' + str(summary.get('schema_version','?')) + '\n\n')

    # Manufactured solution
    parts.append(_h2('Manufactured Solution'))
    m = summary.get('manufactured')
    if not m or 'status' in m:
        parts.append(_missing())
    else:
        parts.append(_kv('grad_estimated_order', round(m.get('grad_estimated_order', float('nan')), 3)))
        parts.append(_kv('lap_estimated_order', round(m.get('lap_estimated_order', float('nan')), 3)))
        parts.append('\n')

    # Adaptive dt
    parts.append(_h2('Adaptive dt PID'))
    a = summary.get('adaptive_dt')
    if not a or 'status' in a:
        parts.append(_missing())
    else:
        metrics = a.get('metrics', {})
        parts.append(_kv('final_residual', metrics.get('final_residual')))
        parts.append(_kv('dt_coeff_var', round(metrics.get('dt_coeff_var', 0.0), 4)))
        parts.append('\n')

    # Plateau classifier
    parts.append(_h2('Plateau Classifier'))
    p = summary.get('plateau')
    if not p or 'status' in p:
        parts.append(_missing())
    else:
        parts.append(_kv('accuracy', round(p.get('accuracy', 0.0), 3)))
        conf = p.get('confusion', {})
        parts.append(f"- confusion pairs: {sum(conf.values())}, classes: {len(conf)}\n\n")

    # Stencil symmetry
    parts.append(_h2('Stencil Symmetry Audit'))
    s = summary.get('stencil')
    if not s or 'status' in s:
        parts.append(_missing())
    else:
        parts.append(_kv('symmetry_norm', f"{s.get('symmetry_norm'):.3e}"))
        bs = s.get('boundary_stats', {})
        parts.append(_kv('nnz', s.get('nnz')))
        parts.append(_kv('boundary_coeffs', bs.get('boundary')))
        parts.append(_kv('interior_coeffs', bs.get('interior')))
        parts.append('\n')

    # Duplication
    parts.append(_h2('Duplication Similarity'))
    d = summary.get('duplication')
    if not d or 'status' in d:
        parts.append(_missing())
    else:
        pairs = d.get('pairs', [])
        warn = d.get('threshold_warn', 0.85)
        block = d.get('threshold_block', 0.9)
        n_warn = sum(1 for x in pairs if x.get('status')=='warn')
        n_block = sum(1 for x in pairs if x.get('status')=='block')
        parts.append(_kv('warn_threshold', warn))
        parts.append(_kv('block_threshold', block))
        parts.append(_kv('warn_pairs', n_warn))
        parts.append(_kv('block_pairs', n_block))
        top = sorted(pairs, key=lambda x: x.get('similarity',0), reverse=True)[:5]
        for t in top:
            parts.append(f"  - top sim: {t.get('similarity'):.3f} (i={t.get('i')}, j={t.get('j')}, {t.get('status')})\n")
        parts.append('\n')

    # Microbench
    parts.append(_h2('Microbench Kernels'))
    mb = summary.get('microbench')
    if not mb or 'status' in mb:
        parts.append(_missing())
    else:
        res = mb.get('results', [])
        by_kernel = {}
        for r in res:
            k = r.get('kernel')
            by_kernel.setdefault(k, []).append(r)
        for k, arr in by_kernel.items():
            avg = sum(x.get('time_per_cell_us',0.0) for x in arr)/len(arr)
            parts.append(_kv(f'avg_time_per_cell_us[{k}]', round(avg, 6)))
        parts.append('\n')

    # Embedding similarity
    parts.append(_h2('Embedding Similarity'))
    e = summary.get('embedding')
    if not e or 'status' in e:
        parts.append(_missing())
    else:
        mode = e.get('mode')
        pairs = e.get('pairs', [])
        parts.append(_kv('mode', mode))
        parts.append(_kv('pairs', len(pairs)))
        top = sorted(pairs, key=lambda x: x.get('similarity',0), reverse=True)[:5]
        for t in top:
            parts.append(f"  - top sim: {t.get('similarity'):.3f} (i={t.get('i')}, j={t.get('j')}, {t.get('status')})\n")
        parts.append('\n')

    # Failure injection
    parts.append(_h2('Failure Injection'))
    fi = summary.get('failure_injection')
    if not fi or 'status' in fi:
        parts.append(_missing())
    else:
        parts.append(_kv('accuracy', round(fi.get('accuracy',0.0),3)))
        parts.append(_kv('runs', fi.get('total')))
        parts.append('\n')

    # Multi-objective tuner
    parts.append(_h2('Multi-Objective Tuner'))
    mo = summary.get('multi_objective')
    if not mo or 'status' in mo:
        parts.append(_missing())
    else:
        parts.append(_kv('pareto_count', mo.get('pareto_count')))
        parts.append(_kv('total', mo.get('total')))
        parts.append('\n')

    return ''.join(parts)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if not os.path.isfile(IN_PATH):
        with open(OUT_PATH,'w',encoding='utf-8') as f:
            f.write('# Playground Dashboard Summary\n\n- status: dashboard JSON missing\n')
        print(json.dumps({'status':'missing'}))
        return
    with open(IN_PATH,'r',encoding='utf-8') as f:
        summary=json.load(f)
    md = render(summary)
    with open(OUT_PATH,'w',encoding='utf-8') as f:
        f.write(md)
    print(json.dumps({'status':'ok','output':OUT_PATH}))

if __name__=='__main__':
    main()
