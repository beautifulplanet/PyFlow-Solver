from __future__ import annotations
from ..core.ghost_fields import State, interior_view

def apply_velocity_bc_cavity(state: State, lid_velocity: float, keep_corners: bool | None = None) -> None:
    u = state.fields['u']; v = state.fields['v']
    ui = interior_view(u); vi = interior_view(v)
    ui[:,0] = 0.0; ui[:,-1] = 0.0; vi[:,0] = 0.0; vi[:,-1] = 0.0
    ui[0,:] = 0.0; vi[0,:] = 0.0
    if ui.shape[1] > 2:
        ui[-1,1:-1] = lid_velocity
    vi[-1,:] = 0.0
    if keep_corners is None:
        keep_corners = bool(state.meta.get('keep_lid_corners', False))
    if keep_corners:
        ui[-1,0] = lid_velocity; ui[-1,-1] = lid_velocity
    else:
        ui[-1,0] = 0.0; ui[-1,-1] = 0.0
    # pressure ghost mirror
    p = state.fields['p']; pi = interior_view(p)
    p[0,1:-1] = pi[0,:]; p[-1,1:-1] = pi[-1,:]
    p[1:-1,0] = pi[:,0]; p[1:-1,-1] = pi[:,-1]

__all__ = ['apply_velocity_bc_cavity']
