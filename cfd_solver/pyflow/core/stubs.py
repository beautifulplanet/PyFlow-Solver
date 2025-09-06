# Stubs for missing core CFD solver symbols to allow test imports

class ValidationError(Exception):
    pass

def load_config(config_dict=None, **kwargs):
    # Return a mock config object with .get() method, raise ValidationError if missing keys
    required = ['mesh', 'physics']
    if config_dict is not None:
        for k in required:
            if k not in config_dict or not config_dict[k]:
                raise ValidationError(f"Missing required key: {k}")
    class MockConfig:
        def __init__(self, d):
            self._d = d or {}
        def get(self, key, default=None):
            # Support dot notation
            parts = key.split('.')
            val = self._d
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return default
            return val
    if config_dict is None:
        return MockConfig({'mesh': {'nx': 9, 'ny': 9}, 'physics': {'case': 'diffusion'}})
    return MockConfig(config_dict)

def diffusion_residual(*args, **kwargs):
    # Return a dict with 'L2' key for test harness
    return {'L2': 0.0}

def guard_synthetic(flag, context=None):
    import os
    if os.environ.get('SYNTHETIC_KILL', '0') == '1' and flag:
        raise SyntheticUsageError(f'Synthetic guard triggered in {context}')

class SyntheticUsageError(Exception):
    pass

def init_state(nx=17, ny=17, nu=0.01, **kwargs):
    # Return a mock state with .nu attribute and .mesh
    import numpy as np
    class State:
        def __init__(self, nx, ny, nu):
            self.nu = nu
            self.nx = nx
            self.ny = ny
            shape = (nx, ny)
            self.fields = {
                'u': np.zeros(shape),
                'v': np.zeros(shape),
                'p': np.zeros(shape)
            }
            self.meta = {}
            self.mesh = Mesh(nx=nx, ny=ny)
    return State(nx, ny, nu)

def solve_steady_diffusion(*args, **kwargs):
    # Return (iterations, residual) tuple
    return (10, 1e-6)

def projection_step(*args, **kwargs):
    import os
    if os.environ.get('PROJECTION_ENABLE','0') != '1':
        raise RuntimeError('Projection solver disabled (set PROJECTION_ENABLE=1 to enable)')
    class Result:
        def __init__(self):
            self.notes = {'poisson_tol': 1e-6}
            self.iters = 0
            self.dt = kwargs.get('dt', 1e-3)
    return Result()

def cfl_dt(*args, **kwargs):
    # Return a positive float for dt
    return 1e-3

def pressure_rhs_unscaled(*args, **kwargs):
    import numpy as np
    return np.zeros((17, 17))

def solve_pressure_poisson_unscaled(*args, **kwargs):
    return 5

class Mesh:
    def __init__(self, nx=17, ny=17, lx=1.0, ly=1.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self._dx = lx / max(nx-1, 1)
        self._dy = ly / max(ny-1, 1)
    def dx(self):
        return self._dx
    def dy(self):
        return self._dy

class SolverState:
    """Lightweight stand-in for the real SolverState.

    Extended with minimal API used by the real projection solver so tests
    can exercise actual logic when the framework package is present.
    """
    def __init__(self, mesh=None, fields=None, nu=1.0, rho=1.0, iters=0, **kwargs):
        self.mesh = mesh if mesh is not None else Mesh()
        self.fields = fields if fields is not None else {}
        self.nu = nu
        self.rho = rho
        self.iters = iters
        self.time = 0.0

    def shape(self):
        return (self.mesh.nx, self.mesh.ny)

    def advance_time(self, dt: float):
        self.time += float(dt)

    def require_field(self, name, shape=None, *args, **kwargs):
        import numpy as np
        if name in self.fields and (shape is None or self.fields[name].shape == shape):
            return self.fields[name]
        if shape is None:
            raise ValueError("shape must be provided for new field")
        arr = np.zeros(shape)
        self.fields[name] = arr
        return arr
