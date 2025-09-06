import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / 'build' / 'hpf_cfd.exe'
OUTFILE = ROOT / 'output' / 'results.dat'
GHIA_FILE = ROOT / 'validation' / 'ghia_re100.dat'


def run_solver(rebuild=False):
    if rebuild or not EXEC.exists():
        # call make from a POSIX-like shell environment; assume user already built if necessary
        print('Executable not found; please build with make in MSYS2 shell if this fails.')
    print('Running solver...')
    subprocess.run([str(EXEC)], check=True)


def load_results():
    data = []
    with open(OUTFILE) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            i,j,u,v,p = parts
            # Fortran output may write indices as floats (e.g., 1.000000)
            i = int(float(i))
            j = int(float(j))
            data.append((i, j, float(u), float(v), float(p)))
    arr = np.array(data)
    nx = int(arr[:,0].max())
    ny = int(arr[:,1].max())
    u = np.zeros((nx,ny))
    v = np.zeros((nx,ny))
    for (i,j,uu,vv,pp) in arr:
        u[int(i)-1,int(j)-1] = uu
        v[int(i)-1,int(j)-1] = vv
    return u, v


def load_ghia():
    lines = GHIA_FILE.read_text().splitlines()
    section = 'u'
    yu, uvals, xv, vvals = [], [], [], []
    for line in lines:
        line=line.strip()
        if not line or line.startswith('#'):
            if 'v_centerline' in line:
                section = 'v'
            continue
        y_or_x, val = map(float, line.split())
        if section == 'u':
            yu.append(y_or_x)
            uvals.append(val)
        else:
            xv.append(y_or_x)
            vvals.append(val)
    return np.array(yu), np.array(uvals), np.array(xv), np.array(vvals)


def extract_centerlines(u, v):
    nx, ny = u.shape
    ic = nx//2
    jc = ny//2
    u_center = u[ic, :]
    v_center = v[:, jc]
    y = np.linspace(0,1,ny)
    x = np.linspace(0,1,nx)
    return x, y, u_center, v_center


def compare():
    u, v = load_results()
    yu, uref, xv, vref = load_ghia()
    x, y, ucl, vcl = extract_centerlines(u, v)
    # Interpolate numerical onto Ghia points
    u_num = np.interp(yu, y, ucl)
    v_num = np.interp(xv, x, vcl)
    err_u = np.sqrt(np.mean((u_num - uref)**2))
    err_v = np.sqrt(np.mean((v_num - vref)**2))
    print(f'Centerline RMSE: u={err_u:.4e} v={err_v:.4e}')
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(u_num, yu, 'r-', label='Numerical')
    ax1.plot(uref, yu, 'ko', ms=4, label='Ghia1982')
    ax1.set_xlabel('u'); ax1.set_ylabel('y'); ax1.set_title('Vertical centerline u(x=0.5)')
    ax1.legend()
    ax2.plot(xv, v_num, 'b-', label='Numerical')
    ax2.plot(xv, vref, 'ks', ms=4, label='Ghia1982')
    ax2.set_xlabel('x'); ax2.set_ylabel('v'); ax2.set_title('Horizontal centerline v(y=0.5)')
    ax2.legend()
    fig.tight_layout()
    out = ROOT / 'output' / 'validation_centerlines.png'
    fig.savefig(out, dpi=150)
    print('Saved plot to', out)
    # Simple pass/fail heuristic
    if err_u < 5e-2 and err_v < 5e-2:
        print('VALIDATION PASS (provisional thresholds).')
    else:
        print('Validation not yet within tolerance (expected at later refinement).')


def main():
    run_solver(False)
    compare()

if __name__ == '__main__':
    main()
