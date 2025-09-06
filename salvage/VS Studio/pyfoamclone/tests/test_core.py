import numpy as np
from pyfoamclone.core.field import Field
from pyfoamclone.numerics.operators import laplacian2d

from pyfoamclone.core.ghost_fields import allocate_state, interior_view, State

def test_field_basic():
    data = np.zeros((4, 4))
    f = Field(data, name='pressure')
    assert f.data.shape == (4, 4)
    assert f.name == 'pressure'
    assert f.location == 'cell_center'
    f.set_boundary('FixedValue', 1.0, 'left')
    assert f.get_boundary('left') == ('FixedValue', 1.0)
    f2 = f.copy()
    assert np.array_equal(f2.data, f.data)
    assert f2.bc == f.bc

def test_laplacian2d_constant_zero():
    import numpy as np
    arr = np.ones((6,6)) * 3.14
    lap = laplacian2d(arr, dx=0.1, dy=0.2)
    assert np.allclose(lap, 0.0)

def test_laplacian2d_quadratic():
    import numpy as np
    nx, ny = 20, 20
    dx = dy = 0.05
    x = np.linspace(0, dx*(nx-1), nx)
    y = np.linspace(0, dy*(ny-1), ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # f = x^2 + y^2 => laplacian = 2 + 2 = 4
    f = X**2 + Y**2
    lap = laplacian2d(f, dx, dy)
    interior = lap[2:-2,2:-2]
    assert np.allclose(interior.mean(), 4.0, rtol=0.05, atol=0.2)

    state = allocate_state(nx, ny, fields=['u','v','p'])
    assert isinstance(state, State)
    assert 'u' in state.fields and 'v' in state.fields and 'p' in state.fields
    assert interior_view(state.fields['u']).shape == (ny, nx)
