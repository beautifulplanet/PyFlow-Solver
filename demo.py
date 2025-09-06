from __future__ import annotations
"""PyFlow v1.0 Demo Script

Runs a short lid-driven cavity simulation with:
  * 64x64 grid
  * 60 steps
  * Checkpoint every 30 steps
  * Structured JSONL logging
Prints a concise summary of final residuals and pressure iterations.
"""
import json, os, sys, time
from pyflow.drivers.simulation_driver import SimulationDriver
from pyflow.core.ghost_fields import allocate_state, interior_view
from pyflow.residuals.manager import ResidualManager
from types import SimpleNamespace

CFG = SimpleNamespace(
    nx=64, ny=64,
    Re=100.0,
    lid_velocity=1.0,
    cfl_target=0.5,
    cfl_growth=1.05,
    advection_scheme='quick',
    disable_advection=False,
    lin_tol=1e-10,
    lin_maxiter=400,
    diagnostics=False,
    enable_jacobi_pc=True,
    force_quiet=True,
    lx=63, ly=63,
)

LOG_PATH = 'run.jsonl'
CK_PATH = 'demo_checkpoint.npz'
setattr(CFG, 'log_path', LOG_PATH)
setattr(CFG, 'emergency_checkpoint_path', CK_PATH)

state = allocate_state(CFG.nx, CFG.ny)
tracker = ResidualManager()

driver = SimulationDriver(CFG, state, tracker)
print('[demo] Starting simulation...')
start = time.time()
final_diag = None
final_residuals = None
for st, residuals, diag in driver.run(max_steps=60, checkpoint_path=CK_PATH, checkpoint_interval=30):
  final_diag = diag
  final_residuals = residuals
print('[demo] Completed in %.2fs' % (time.time() - start))
if final_diag and final_residuals:
  print('[demo] Final continuity=%.3e Rp_it=%s' % (final_residuals['continuity'], final_diag.get('Rp_iterations')))
  uc = final_diag.get('u_centerline', [[], []])
  print('[demo] Sample centerline u(y) length:', len(uc[0]) if uc and len(uc) == 2 else 0)
else:
  print('[demo] No steps executed.')
print('[demo] Log written to', LOG_PATH)
print('[demo] Checkpoint at', CK_PATH)
