from __future__ import annotations
import numpy as np, pathlib

def save_vtk(state, filename: str):
    """Write simple structured grid VTK (legacy ASCII) with u,v,p scalar fields."""
    path = pathlib.Path(filename)
    u = state.fields.get('u'); v = state.fields.get('v'); p = state.fields.get('p')
    nx, ny = u.shape
    # Legacy VTK structured points
    with path.open('w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('CFD state\nASCII\nDATASET STRUCTURED_POINTS\n')
        f.write(f'DIMENSIONS {nx} {ny} 1\n')
        f.write('ORIGIN 0 0 0\n')
        f.write('SPACING 1 1 1\n')
        f.write(f'POINT_DATA {nx*ny}\n')
        def write_field(name, arr):
            f.write(f'SCALARS {name} float 1\nLOOKUP_TABLE default\n')
            flat = arr.reshape(-1)
            for val in flat:
                f.write(f"{float(val)}\n")
        write_field('u', u)
        write_field('v', v)
        write_field('p', p)
    return str(path)
