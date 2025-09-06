# PyFlow Hybrid Solver User Guide

This guide explains how to use the hybrid Python/C++ implementation of the PyFlow CFD solver. The hybrid approach combines the flexibility of Python with the performance of C++, resulting in significantly faster simulations without sacrificing ease of use.

## Installation

### Prerequisites

- Python 3.7 or newer
- NumPy
- Matplotlib
- pybind11
- A C++ compiler:
  - Windows: Visual Studio 2019 or newer with C++ desktop development workload
  - Linux: GCC 7 or newer
  - macOS: Clang (via Xcode Command Line Tools)

### Installation Steps

1. Install required Python packages:

```bash
pip install numpy matplotlib pybind11
```

2. Build the C++ extensions:

#### Windows
```
build_extensions.bat
```

#### Linux/macOS
```bash
python setup.py build_ext --inplace
```

3. Verify installation:

```bash
python -c "import pyflow_core; print('PyFlow C++ core extension successfully installed!')"
```

## Basic Usage

### Running a Simple Simulation

```python
from pyflow.hybrid_solver import solve_lid_driven_cavity
from pyflow.grid import Grid
from pyflow.logging import LiveLogger
import numpy as np
import matplotlib.pyplot as plt

# Set problem parameters
N = 65          # Grid size
L = 1.0         # Domain length
Re = 100        # Reynolds number
dt = 0.001      # Time step
T = 5.0         # Total simulation time

# Create grid
grid = Grid(N, L)
dx = dy = grid.dx

# Create logger
logger = LiveLogger(N, Re, dt, T, log_interval=100)

# Solve the problem using the hybrid solver
u, v, p, residuals = solve_lid_driven_cavity(
    N, dx, dy, Re, dt, T, 
    p_iterations=500,
    logger=logger
)

# Plot results
plt.figure(figsize=(10, 8))
plt.contourf(grid.X, grid.Y, u, levels=20, cmap='viridis')
plt.colorbar(label='u-velocity')
plt.streamplot(grid.X.T, grid.Y.T, u.T, v.T, color='white', density=1)
plt.title(f'Lid-Driven Cavity Flow at Re={Re}')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(f'lid_driven_cavity_Re{Re}.png', dpi=300)
plt.show()

# Plot residuals
plt.figure(figsize=(12, 4))
plt.semilogy(residuals['u_res'], label='u-momentum')
plt.semilogy(residuals['v_res'], label='v-momentum')
plt.semilogy(residuals['cont_res'], label='continuity')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.title('Convergence History')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'residuals_Re{Re}.png', dpi=300)
plt.show()
```

## Advanced Usage

### Customizing Solver Parameters

```python
from pyflow.hybrid_solver import solve_lid_driven_cavity

# Advanced parameters
u, v, p, residuals = solve_lid_driven_cavity(
    N=65,
    dx=0.01,
    dy=0.01,
    Re=400,
    dt=0.0005,
    T=10.0,
    p_iterations=1000,     # Number of pressure solver iterations
    alpha_u=0.7,           # Under-relaxation factor for velocity
    alpha_p=0.3            # Under-relaxation factor for pressure
)
```

### Running Benchmarks

To compare the performance of the pure Python and hybrid implementations:

```bash
python benchmark_hybrid.py
```

You can customize the benchmark parameters:

```bash
# Run specific Reynolds numbers
python benchmark_hybrid.py --re 100,400

# Run specific grid sizes
python benchmark_hybrid.py --grid 33,65,97

# Combine parameters
python benchmark_hybrid.py --re 100,400 --grid 33,65
```

## Performance Considerations

### When to Use the Hybrid Solver

The hybrid solver provides the most benefit in these scenarios:

1. **Large grid sizes**: The speedup is most significant for grids larger than 65×65
2. **High Reynolds numbers**: Simulations requiring many time steps benefit most
3. **Production runs**: When you've finalized your setup and need the fastest execution

### Performance Tips

1. **Adjust pressure iterations**: Higher values improve accuracy but reduce speed
   ```python
   solve_lid_driven_cavity(..., p_iterations=1000)  # More accurate but slower
   solve_lid_driven_cavity(..., p_iterations=100)   # Faster but less accurate
   ```

2. **Set appropriate under-relaxation**: Lower values for high Reynolds numbers
   ```python
   # For Re=100
   solve_lid_driven_cavity(..., alpha_u=0.8, alpha_p=0.5)
   
   # For Re=1000
   solve_lid_driven_cavity(..., alpha_u=0.5, alpha_p=0.2)
   ```

3. **Optimize logging frequency**: Reduce for faster execution
   ```python
   logger = LiveLogger(N, Re, dt, T, log_interval=1000)  # Log less frequently
   ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'pyflow_core'**
   - The C++ extensions have not been built successfully
   - Solution: Run the build script again and check for errors

2. **"Not using C++ extensions" warning**
   - The solver is falling back to pure Python implementation
   - Solution: Ensure the C++ extensions are properly built

3. **Simulation diverges at high Reynolds numbers**
   - Reduce the time step
   - Increase under-relaxation (lower alpha values)
   - Increase grid resolution

4. **Slow performance despite using hybrid solver**
   - Check that the C++ extensions are actually being used
   - Try increasing the problem size (C++ benefit increases with problem size)
   - Check your compiler optimization settings

## API Reference

### `solve_lid_driven_cavity`

```python
def solve_lid_driven_cavity(
    N, dx, dy, Re, dt, T, 
    p_iterations=500, 
    alpha_u=None, 
    alpha_p=None, 
    logger=None
):
    """
    Solve the lid-driven cavity flow problem using the hybrid solver
    
    Parameters:
    -----------
    N : int
        Number of grid points in each direction
    dx, dy : float
        Grid spacing in x and y directions
    Re : float
        Reynolds number
    dt : float
        Time step size
    T : float
        Total simulation time
    p_iterations : int, optional
        Number of iterations for pressure Poisson solver
    alpha_u : float, optional
        Under-relaxation factor for velocity
        If None, an adaptive value based on Re is used
    alpha_p : float, optional
        Under-relaxation factor for pressure
        If None, an adaptive value based on Re is used
    logger : LiveLogger, optional
        Logger object for monitoring progress
    
    Returns:
    --------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    residuals : dict
        Dictionary with residual histories
    """
```

For detailed API documentation, refer to the docstrings in the code.

## Contributing

Contributions to improve the solver are welcome! Areas that could benefit from further development:

1. **More C++ optimizations**: Additional performance-critical functions
2. **Parallel computing**: Multi-threading for larger problems
3. **Additional physics**: Heat transfer, turbulence models, etc.
4. **Improved visualization**: Interactive plotting and animation
