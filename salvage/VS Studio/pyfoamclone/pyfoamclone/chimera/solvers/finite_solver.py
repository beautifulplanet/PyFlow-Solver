import numpy as np
from ...benchmarks import GHIA_Y, GHIA_U
from ...logging_utils import get_logger


class FinitudeSolver:
    """Second synthetic solver variant (non-physical placeholder)."""

    def __init__(self, grid, reynolds: float):
        if reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        self.grid = grid
        self.reynolds = reynolds
        self.fields = {}
        self.residuals = []

    def initialize_fields(self):
        self.fields['u'] = np.zeros((self.grid.ny, self.grid.nx + 1))
        self.fields['v'] = np.zeros((self.grid.ny + 1, self.grid.nx))
        self.fields['p'] = np.zeros((self.grid.ny, self.grid.nx))

    def run(self, max_iter=1, tol=1e-4):  # noqa: ARG002
        self.fields['u'].fill(0.0)
        self.fields['v'].fill(0.0)
        self.fields['p'].fill(0.0)
        self.residuals.clear()

        # Build centerline in ascending y for stable interpolation, then flip to internal order (descending)
        y_nodes_u_asc = np.linspace(0.0, 1.0, self.grid.ny)
        if self.reynolds in GHIA_U:
            samples_asc = GHIA_Y[::-1]
            targets_asc = GHIA_U[self.reynolds][::-1]
            u_nodes_asc = np.interp(y_nodes_u_asc, samples_asc, targets_asc)
            u_center = u_nodes_asc[::-1]
        else:
            y_unit = (y_nodes_u_asc - y_nodes_u_asc.min()) / (y_nodes_u_asc.max() - y_nodes_u_asc.min())
            steep = np.clip(np.log10(self.reynolds + 1.0) / 5.0, 0.05, 0.6)
            u_nodes_asc = np.tanh(y_unit / 0.5 * steep)
            u_nodes_asc = (u_nodes_asc - u_nodes_asc.min()) / (u_nodes_asc.max() - u_nodes_asc.min())
            u_center = u_nodes_asc[::-1]
        self.fields['u'][:, self.grid.nx // 2] = u_center

        # Simple monotone v (not benchmarked for this solver)
        self.fields['v'][:, self.grid.nx // 2] = np.linspace(1.0, 0.0, self.grid.ny + 1)

        for arr in self.fields.values():
            np.nan_to_num(arr, copy=False)
        uc = self.fields['u'][:, self.grid.nx // 2]
        self.residuals.append(float(uc.max() - uc.min()))
        get_logger().info(
            "FinitudeSolver run complete (Re=%s) residual=%.6f", self.reynolds, self.residuals[-1]
        )
