import os, numpy as np, types
from pyfoamclone.io.vtk_writer import save_vtk

class DummyState:
    def __init__(self, nx=8, ny=8):
        self.fields = {
            'u': np.zeros((ny, nx)),
            'v': np.zeros((ny, nx)),
            'p': np.zeros((ny, nx))
        }

def test_vtk_writer_creates_file(tmp_path):
    st = DummyState()
    fname = tmp_path / 'state.vtk'
    out = save_vtk(st, str(fname))
    assert os.path.exists(out)
    text = open(out,'r').read().splitlines()[0]
    assert 'vtk DataFile' in text