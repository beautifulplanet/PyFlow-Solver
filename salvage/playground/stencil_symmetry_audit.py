"""Playground: Stencil Symmetry & Boundary Audit (v0.1)
Generates a sparse Laplacian matrix for a 2D grid (5-point), audits symmetry and boundary coefficient patterns.
"""
from __future__ import annotations
import argparse, json, os, math
import numpy as np

SCHEMA_VERSION=1

def laplacian_matrix(nx, ny):
    N = nx*ny
    row=[]; col=[]; data=[]
    def idx(i,j): return i*nx + j
    for i in range(ny):
        for j in range(nx):
            center = idx(i,j)
            coeff=-4.0
            row.append(center); col.append(center); data.append(coeff)
            if j>0:
                row.append(center); col.append(idx(i,j-1)); data.append(1.0)
            if j<nx-1:
                row.append(center); col.append(idx(i,j+1)); data.append(1.0)
            if i>0:
                row.append(center); col.append(idx(i-1,j)); data.append(1.0)
            if i<ny-1:
                row.append(center); col.append(idx(i+1,j)); data.append(1.0)
    return np.array(row), np.array(col), np.array(data), N

def symmetry_norm(row,col,data,N):
    from collections import defaultdict
    d=defaultdict(float)
    for r,c,v in zip(row,col,data):
        d[(r,c)] += v
    asym=0.0; total=0.0
    for (r,c),v in d.items():
        vt = d.get((c,r),0.0)
        diff = v - vt
        asym += diff*diff
        total += v*v + vt*vt
    return math.sqrt(asym)/math.sqrt(total) if total else 0.0

def boundary_coeff_stats(row,col,data,nx,ny):
    from collections import defaultdict
    counts={'interior':0,'boundary':0}
    for r,c,v in zip(row,col,data):
        i = r//nx; j = r%nx
        if i==0 or j==0 or i==ny-1 or j==nx-1:
            counts['boundary']+=1
        else:
            counts['interior']+=1
    return counts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--ny', type=int, default=32)
    ap.add_argument('--output-dir', default='playground/artifacts/stencil')
    args=ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    row,col,data,N=laplacian_matrix(args.nx,args.ny)
    sym = symmetry_norm(row,col,data,N)
    stats = boundary_coeff_stats(row,col,data,args.nx,args.ny)
    out={'schema_version':SCHEMA_VERSION,'nx':args.nx,'ny':args.ny,'symmetry_norm':sym,'boundary_stats':stats,'nnz':int(len(data))}
    with open(os.path.join(args.output_dir,'stencil_audit.json'),'w') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out))

if __name__=='__main__':
    main()
