"""Lid-driven cavity benchmark using the reusable SimulationDriver.

This script now delegates the core loop to `SimulationDriver`, yielding
step payloads until steady-state (or max steps) is reached. It remains a
simple executable reference case and a smoke test for downstream UI layers.
"""
from __future__ import annotations
import numpy as np
from src.pyflow.driver.simulation_driver import (
    SimulationDriver, DriverConfig, PhysicsConfig
)
from src.pyflow.core.ghost_fields import interior_view

def main():
    physics = PhysicsConfig(
        Re=100.0,
        lid_velocity=1.0,
        advection_scheme='quick',
        diagnostics=False,
    )
    driver_cfg = DriverConfig(
        nx=65,
        ny=65,
        max_steps=10_000,
        report_interval=100,
        steady_window=100,
        steady_slope_tol=2e-3,
        stop_on_steady=True,
    )
    driver = SimulationDriver(driver_cfg, physics)
    print(f"--- Lid-Driven Cavity Start ---\nGrid: {driver_cfg.nx}x{driver_cfg.ny} Re={physics.Re} lidU={physics.lid_velocity}\n")
    last_payload = None
    for payload in driver.run():
        last_payload = payload
        if payload['steady'] and driver_cfg.stop_on_steady:
            break
    if last_payload is None:
        print("No iterations performed.")
        return
    state = last_payload['state']
    ui = interior_view(state.fields['u'])
    vi = interior_view(state.fields['v'])
    u_centerline = ui[:, ui.shape[1]//2]
    v_centerline = vi[vi.shape[0]//2, :]
    print("\n--- Simulation Complete ---")
    print(f"Elapsed wall time: {driver.elapsed:.2f} s, iterations: {last_payload['iteration']+1}")
    if last_payload['steady']:
        print(f"Steady-state detected at iteration {driver.steady_iteration}.")
    else:
        print("Steady-state not detected within iteration budget.")
    res = last_payload['residuals']
    print(f"Final Ru={res['Ru']:.3e} Rv={res['Rv']:.3e}")
    print(
        f"Final mean divergence={last_payload['divergence_mean']:.3e}, "
        f"mean-free divergence norm={last_payload['divergence_mean_free_norm']:.3e}"
    )
    print("Sample centerline velocity profiles (u mid-x, v mid-y):")
    print("u(y) centerline:", np.array2string(u_centerline, precision=4, suppress_small=True))
    print("v(x) centerline:", np.array2string(v_centerline, precision=4, suppress_small=True))
    print("--- End ---")

if __name__ == "__main__":  # pragma: no cover
    main()
