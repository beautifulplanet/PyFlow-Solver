import numpy as np
from pyflow.core.field import Field

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
