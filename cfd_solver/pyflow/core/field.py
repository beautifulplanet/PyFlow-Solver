import numpy as np

class Field:
    """
    Represents a physical field (e.g., velocity, pressure) on the mesh.
    Wraps a NumPy array and stores metadata such as name, location, and boundary conditions.
    """
    def __init__(self, data, name, location='cell_center', bc=None):
        self.data = np.array(data)
        self.name = name
        self.location = location  # 'cell_center', 'face', etc.
        self.bc = bc or {}

    def set_boundary(self, bc_type, value, face):
        self.bc[face] = (bc_type, value)

    def get_boundary(self, face):
        return self.bc.get(face, None)

    def copy(self):
        return Field(self.data.copy(), self.name, self.location, self.bc.copy())
