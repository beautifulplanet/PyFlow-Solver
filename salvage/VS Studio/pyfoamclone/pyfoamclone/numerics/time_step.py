from __future__ import annotations


def compute_dt(cfl_target: float, current_cfl: float, prev_dt: float, max_growth: float = 1.1):
    if current_cfl <= 0:
        return prev_dt
    factor = cfl_target / current_cfl
    # limit growth
    if factor > max_growth:
        factor = max_growth
    return prev_dt * factor
