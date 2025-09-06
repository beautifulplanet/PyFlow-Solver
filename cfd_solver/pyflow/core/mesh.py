from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass(slots=True)
class GridSpec:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def dx_dy(self) -> Tuple[float, float]:
        return self.lx / self.nx, self.ly / self.ny

def build_grid(spec: GridSpec):
    dx, dy = spec.dx_dy()
    xc = (np.arange(spec.nx) + 0.5) * dx
    yc = (np.arange(spec.ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    return {
        "xc": xc,
        "yc": yc,
        "Xc": Xc,
        "Yc": Yc,
        "dx": dx,
        "dy": dy,
        "nx": spec.nx,
        "ny": spec.ny,
    }
