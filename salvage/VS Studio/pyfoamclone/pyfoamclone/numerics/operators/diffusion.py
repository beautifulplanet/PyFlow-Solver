import numpy as np

def laplacian2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute a simple 2D Laplacian with homogeneous Neumann at boundary (second-order interior, copy edges).

    Parameters
    ----------
    field : np.ndarray (ny, nx)
    dx, dy : float
    Returns
    -------
    np.ndarray : Laplacian approximation same shape as field.
    """
    ny, nx = field.shape
    lap = np.zeros_like(field)
    # interior
    lap[1:-1,1:-1] = (
        (field[1:-1,2:] - 2*field[1:-1,1:-1] + field[1:-1,0:-2]) / dx**2 +
        (field[2:,1:-1] - 2*field[1:-1,1:-1] + field[0:-2,1:-1]) / dy**2
    )
    # crude copy edges (could improve later)
    lap[0,:] = lap[1,:]
    lap[-1,:] = lap[-2,:]
    lap[:,0] = lap[:,1]
    lap[:,-1] = lap[:,-2]
    return lap
