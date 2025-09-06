import os, math
import numpy as np
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import pressure_rhs_unscaled, solve_pressure_poisson_unscaled


def make_state(n=33):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.01)
    u = st.require_field('u', (n,n))
    v = st.require_field('v', (n,n))
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u[:,:] = np.sin(math.pi*X)
    v[:,:] = np.cos(math.pi*Y)
    st.require_field('p', (n,n))
    return st


def solve_with(method):
    os.environ['PROJECTION_LINSOLVER'] = method
    os.environ['PROJECTION_ENABLE'] = '1'
    st = make_state()
    rhs = pressure_rhs_unscaled(st)
    iters = solve_pressure_poisson_unscaled(rhs, st, tol=1e-5, max_iter=6000)
    return iters


def test_sor_not_slower_than_jacobi():
    jacobi_iters = solve_with('jacobi')
    sor_iters = solve_with('sor')
    # SOR should converge in fewer iterations typically; allow small tolerance
    assert sor_iters <= jacobi_iters * 0.9 + 5, f"SOR not faster: jacobi {jacobi_iters} sor {sor_iters}"
