import numpy as np

def apply_inlet(u, velocity):
    """
    Apply inlet boundary condition to u-velocity field
    """
    u[:, 0] = velocity

def apply_no_slip_walls(u, v):
    """
    Apply no-slip boundary conditions to velocity fields
    """
    # Top and bottom walls for u
    u[0, :] = 0
    u[-1, :] = 0
    # Left and right walls for v
    v[:, 0] = 0
    v[:, -1] = 0
