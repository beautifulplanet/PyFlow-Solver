"""Fast Checks Runner (v0.1)
Runs a short subset of playground scripts to produce the dashboard quickly (CI-friendly).
"""
from __future__ import annotations
import subprocess, sys

CMDS=[
    [sys.executable, 'playground/manufactured_solution_order.py', '--grids', '17', '33', '65'],
    [sys.executable, 'playground/residual_plateau_classifier_demo.py', '--seeds', '20', '--steps', '200'],
    [sys.executable, 'playground/microbench_kernels.py', '--sizes', '64', '128', '--reps', '20'],
    [sys.executable, 'playground/aggregate_dashboard.py'],
    [sys.executable, 'playground/render_dashboard_md.py'],
]

def main():
    for cmd in CMDS:
        print('>',' '.join(cmd))
        subprocess.check_call(cmd)

if __name__=='__main__':
    main()
