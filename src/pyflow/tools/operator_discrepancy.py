from __future__ import annotations

"""Operator Discrepancy Analysis Tool (Phase 3 Diagnostics)

Compares the legacy assembled pressure matrix (negative Laplacian with
identity boundary rows) against a matrix reconstructed from the matrix-free
composition L = div(grad()).

Outputs a structured discrepancy report to stdout for a given grid.

Usage (from repo root):
    python -m pyflow.tools.operator_discrepancy --nx 16 --ny 16 --dx 1 --dy 1
"""
import argparse

import numpy as np
import scipy.sparse as sp

from pyflow.numerics.mf_ops import build_laplacian_matrix_from_ops
from pyflow.solvers.pressure_solver import assemble_negative_laplacian


def build_matrix_free_equivalent(nx:int, ny:int, dx:float, dy:float) -> sp.csr_matrix:
    """Build matrix equivalent to legacy assemble_negative_laplacian using mf ops.

    Steps:
      1. Build L (Laplacian) via composition.
      2. Convert to -L to match legacy sign (legacy stores -∇²).
      3. Overwrite all boundary rows with identity rows (Dirichlet-like) to match
         legacy pressure matrix treatment (stabilizing but non-physical).
    """
    L = build_laplacian_matrix_from_ops(nx, ny, dx, dy)  # Laplacian
    A = (-L).tocsr()  # negate to get -Laplace
    A = A.tolil()
    def is_boundary(i,j):
        return i==0 or j==0 or i==nx-1 or j==ny-1
    for j in range(ny):
        for i in range(nx):
            if is_boundary(i,j):
                k = j*nx + i
                A.rows[k] = [k]
                A.data[k] = [1.0]
    return A.tocsr()


def analyze(nx:int, ny:int, dx:float, dy:float, tol:float=1e-12):
    A_legacy = assemble_negative_laplacian(nx, ny, dx, dy).tocsr()
    A_mf_raw = build_laplacian_matrix_from_ops(nx, ny, dx, dy).tocsr()  # L
    A_mf_neg = (-A_mf_raw).tocsr()  # -L
    A_mf_equiv = build_matrix_free_equivalent(nx, ny, dx, dy)  # -L with identity boundaries

    def row_type(idx):
        j = idx // nx
        i = idx - j*nx
        if i==0 or j==0 or i==nx-1 or j==ny-1:
            return 'boundary'
        return 'interior'

    # Compute discrepancies
    D_raw = (A_legacy - A_mf_neg).tocsr()        # before boundary identity alignment
    D_equiv = (A_legacy - A_mf_equiv).tocsr()    # after alignment

    # Row-wise norms
    D_raw_row_norms = np.array([np.linalg.norm(D_raw[k].data) for k in range(nx*ny)])
    D_equiv_row_norms = np.array([np.linalg.norm(D_equiv[k].data) for k in range(nx*ny)])

    types = [row_type(k) for k in range(nx*ny)]

    def summarize(norms, label):
        interior = [norms[k] for k,t in enumerate(types) if t=='interior']
        boundary = [norms[k] for k,t in enumerate(types) if t=='boundary']
        return {
            'label': label,
            'interior_max': float(np.max(interior)) if interior else 0.0,
            'interior_mean': float(np.mean(interior)) if interior else 0.0,
            'interior_nonzero_rows': int(np.sum(np.array(interior) > tol)),
            'boundary_max': float(np.max(boundary)) if boundary else 0.0,
            'boundary_mean': float(np.mean(boundary)) if boundary else 0.0,
            'boundary_nonzero_rows': int(np.sum(np.array(boundary) > tol)),
        }

    sum_raw = summarize(D_raw_row_norms, 'legacy - (-L) (raw)')
    sum_equiv = summarize(D_equiv_row_norms, 'legacy - mf_equiv (after boundary identity)')

    # Identify sample mismatching interior rows after equivalence alignment
    offending_rows = [k for k in range(nx*ny) if types[k]=='interior' and D_equiv_row_norms[k] > tol]
    samples = offending_rows[:5]

    report = {
        'grid': {'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy},
        'tolerance': tol,
        'summary_raw': sum_raw,
        'summary_equiv': sum_equiv,
        'offending_interior_rows': offending_rows,
        'offending_interior_rows_sample': samples,
    }
    return report, (A_legacy, A_mf_neg, A_mf_equiv), (D_raw, D_equiv)


def format_report(report):
    lines = []
    lines.append("=== OPERATOR DISCREPANCY REPORT ===")
    g = report['grid']
    lines.append(f"Grid: nx={g['nx']}, ny={g['ny']}, dx={g['dx']}, dy={g['dy']}")
    lines.append(f"Tolerance: {report['tolerance']}")
    for key in ('summary_raw','summary_equiv'):
        s = report[key]
        lines.append(f"-- {s['label']} --")
        lines.append(f"  Interior: max={s['interior_max']:.3e} mean={s['interior_mean']:.3e} nonzero_rows>{report['tolerance']} = {s['interior_nonzero_rows']}")
        lines.append(f"  Boundary: max={s['boundary_max']:.3e} mean={s['boundary_mean']:.3e} nonzero_rows>{report['tolerance']} = {s['boundary_nonzero_rows']}")
    off = report['offending_interior_rows']
    lines.append(f"Interior rows differing after alignment: {len(off)}")
    if off:
        lines.append(f"Sample differing interior rows: {report['offending_interior_rows_sample']}")
    else:
        lines.append("All interior rows match within tolerance after boundary alignment.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=16)
    ap.add_argument('--ny', type=int, default=16)
    ap.add_argument('--dx', type=float, default=1.0)
    ap.add_argument('--dy', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=1e-12)
    args = ap.parse_args()
    report, mats, diffs = analyze(args.nx, args.ny, args.dx, args.dy, args.tol)
    import scipy.sparse as sp
    A_legacy, A_mf_neg, A_mf_equiv = [sp.csr_matrix(m) for m in mats]
    _, D_equiv = diffs
    D_equiv = sp.csr_matrix(D_equiv)
    print(format_report(report))
    # Detailed row dump for first few differing interior rows
    off = report['offending_interior_rows_sample']
    if off:
        print("\n--- Detailed Stencil Differences (legacy vs -L) for sample rows ---")
        for k in off:
            leg_row = A_legacy.getrow(k)
            mf_row = A_mf_neg.getrow(k)
            diff_row = D_equiv.getrow(k)
            def row_to_dict(r):
                return {int(c): float(v) for c,v in zip(r.indices, r.data, strict=False)}
            print(f"Row {k}:")
            print("  legacy:", row_to_dict(leg_row))
            print("  -L    :", row_to_dict(mf_row))
            print("  diff  :", row_to_dict(diff_row))

if __name__ == '__main__':
    main()
