import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_field(u, v, grid):
    """
    Plot the velocity field.
    Args:
        u: x-component of velocity (2D array)
        v: y-component of velocity (2D array)
        grid: Grid object containing mesh information
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, grid.lx, grid.nx+1)
    y = np.linspace(0, grid.ly, grid.ny+1)
    X, Y = np.meshgrid(x, y)
    # Adjust u, v shape if needed for quiver
    u_plot = u if u.shape == X.shape else u[:X.shape[0], :X.shape[1]]
    v_plot = v if v.shape == X.shape else v[:X.shape[0], :X.shape[1]]
    ax.quiver(X, Y, u_plot, v_plot)
    ax.set_title('Velocity Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.close(fig)
    return fig

def plot_pressure_field(p, grid):
    """
    Plot the pressure field.
    Args:
        p: pressure field (2D array)
        grid: Grid object containing mesh information
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, grid.lx, grid.nx+1)
    y = np.linspace(0, grid.ly, grid.ny+1)
    X, Y = np.meshgrid(x, y)
    # Adjust p shape if needed for contourf
    p_plot = p if p.shape == X.shape else p[:X.shape[0], :X.shape[1]]
    contour = ax.contourf(X, Y, p_plot)
    fig.colorbar(contour, ax=ax)
    ax.set_title('Pressure Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.close(fig)
    return fig
