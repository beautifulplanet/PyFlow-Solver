import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless testing
from pyfoamclone.chimera.grid import Grid
from pyfoamclone.chimera.utils.visualization import plot_velocity_field, plot_pressure_field

def test_plot_velocity_field_runs():
	grid = Grid(8, 8, 1.0, 1.0)
	u = np.ones((9, 9))
	v = np.ones((9, 9))
	try:
		plot_velocity_field(u, v, grid)
	except Exception as e:
		assert False, f"plot_velocity_field failed: {e}"

def test_plot_pressure_field_runs():
	grid = Grid(8, 8, 1.0, 1.0)
	p = np.ones((9, 9))
	try:
		plot_pressure_field(p, grid)
	except Exception as e:
		assert False, f"plot_pressure_field failed: {e}"
