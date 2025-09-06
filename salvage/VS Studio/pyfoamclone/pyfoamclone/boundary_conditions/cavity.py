from __future__ import annotations
import numpy as np
from ..core.ghost_fields import State, interior_view

"""Boundary condition helpers for lid-driven cavity."""

def apply_velocity_bc_cavity(state: State, lid_velocity: float, keep_corners: bool | None = None) -> None:
    """Apply no-slip walls and moving lid BC (legacy: corners move with lid).

    Tests for boundary enforcement expect the entire top interior row (including what
    numerically corresponds to corner interior cells) to equal lid_velocity.
    """
    u = state.fields['u']
    v = state.fields['v']
    ui = interior_view(u)
    vi = interior_view(v)
    # Walls no-slip
    ui[:, 0] = 0.0; ui[:, -1] = 0.0
    vi[:, 0] = 0.0; vi[:, -1] = 0.0
    ui[0, :] = 0.0; vi[0, :] = 0.0
    # Lid interior excluding corners (tests expect corners zero)
    if ui.shape[1] > 2:
        ui[-1,1:-1] = lid_velocity
    vi[-1,:] = 0.0
    if keep_corners is None:
        keep_corners = bool(state.meta.get('keep_lid_corners', False))
    if keep_corners:
        ui[-1,0] = lid_velocity
        ui[-1,-1] = lid_velocity
    else:
        ui[-1,0] = 0.0
        ui[-1,-1] = 0.0
    # Diagnostic: print lid row after BC enforcement
    print("Lid row after BC:", ui[-1,:])
    # Mirror pressure to ghosts
    p = state.fields['p']
    pi = interior_view(p)
    p[0,1:-1] = pi[0,:]           # bottom ghost row
    p[-1,1:-1] = pi[-1,:]         # top ghost row
    p[1:-1,0] = pi[:,0]           # left ghost col
    p[1:-1,-1] = pi[:,-1]         # right ghost col
