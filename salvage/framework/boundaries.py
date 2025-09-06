"""Boundary condition helper functions for projection step.

Each function mutates the velocity (and optionally pressure) fields of the
provided SolverState in-place. They are designed to be lightweight callables
passed to projection_step(boundary_fn=...).
"""
from __future__ import annotations
from .state import SolverState
import numpy as np

__all__ = [
    'no_slip_all',
    'periodic_x',
    'free_slip_all',
    'lid_driven_cavity_top',
]

def no_slip_all(state: SolverState):
    u = state.fields['u']; v = state.fields['v']
    u[0,:] = 0.0; u[-1,:] = 0.0; u[:,0] = 0.0; u[:,-1] = 0.0
    v[0,:] = 0.0; v[-1,:] = 0.0; v[:,0] = 0.0; v[:,-1] = 0.0


def periodic_x(state: SolverState):
    """Periodic boundary in x; copy opposite edges for u,v."""
    u = state.fields['u']; v = state.fields['v']
    u[0,:] = u[-2,:]; u[-1,:] = u[1,:]
    v[0,:] = v[-2,:]; v[-1,:] = v[1,:]
    # y boundaries remain whatever previous function set (compose by calling prior)


def free_slip_all(state: SolverState):
    """Free-slip on all walls: zero normal velocity, zero normal derivative of tangential.

    For simplicity: set normal velocity to 0 at boundary and copy adjacent interior
    value for tangential component (Neumann 1st order)."""
    u = state.fields['u']; v = state.fields['v']
    # Bottom/top (normal v):
    v[:,0] = 0.0; v[:,-1] = 0.0
    u[:,0] = u[:,1]; u[:,-1] = u[:,-2]
    # Left/right (normal u):
    u[0,:] = 0.0; u[-1,:] = 0.0
    v[0,:] = v[1,:]; v[-1,:] = v[-2,:]


def lid_driven_cavity_top(state: SolverState, lid_u: float = 1.0):
    """No-slip walls except top lid with constant horizontal velocity lid_u."""
    u = state.fields['u']; v = state.fields['v']
    # No-slip sides + bottom
    u[0,:] = 0.0; u[-1,:] = 0.0; u[:,0] = 0.0
    v[0,:] = 0.0; v[-1,:] = 0.0; v[:,0] = 0.0
    # Top lid
    u[:,-1] = lid_u
    v[:,-1] = 0.0
