import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / 'build' / 'hpf_cfd.exe'
OUTPUT_FILE = ROOT / 'output' / 'results.dat'


def run_solver():
    if not EXEC.exists():
        raise SystemExit('Executable not found. Build with: make')
    print('Running solver...')
    subprocess.run([str(EXEC)], check=True)
    if not OUTPUT_FILE.exists():
        raise SystemExit('Expected output file not found: ' + str(OUTPUT_FILE))


def load_results():
    data = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            i, j = map(int, parts[:2])
            u, v, p = map(float, parts[2:])
            data.append((i, j, u, v, p))
    arr = np.array(data)
    return arr


def plot_field(arr, nx=None, ny=None):
    if nx is None:
        nx = int(arr[:,0].max())
    if ny is None:
        ny = int(arr[:,1].max())
    u = np.zeros((nx, ny))
    for row in arr:
        i, j, uu = int(row[0])-1, int(row[1])-1, row[2]
        u[i,j] = uu
    plt.imshow(u.T, origin='lower', cmap='viridis')
    plt.colorbar(label='u')
    plt.title('Lid-driven cavity u velocity (placeholder)')
    plt.xlabel('i'); plt.ylabel('j')
    plt.tight_layout()
    out_png = OUTPUT_FILE.with_suffix('.png')
    plt.savefig(out_png, dpi=150)
    print('Saved plot to', out_png)


def main():
    run_solver()
    arr = load_results()
    plot_field(arr)

if __name__ == '__main__':
    main()
