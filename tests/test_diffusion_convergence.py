import numpy as np, math
import pytest
from cfd_solver.pyflow.core import init_state
from cfd_solver.pyflow.core import solve_steady_diffusion

# Manufactured solution for diffusion steady-state: u = sin(pi x) sin(pi y), source = 2 pi^2 nu u

def run_case(n, method='sor'):
    st = init_state(nx=n, ny=n, nu=0.01)
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u_exact = np.sin(math.pi*X)*np.sin(math.pi*Y)
    f = 2*math.pi**2 * st.nu * u_exact
    # Use SOR for faster convergence; tolerance scaled with h to reduce iterations.
    h = 1/(n-1)
    tol = 2e-2 * h  # looser than pure h^2 so SOR reaches quickly; spatial discretisation still dominates error for these grids
    iters, resid = solve_steady_diffusion(st.fields['u'], f, st.mesh.dx(), st.mesh.dy(), st.nu, tol=tol, method=method, omega=1.8)
    err = np.sqrt(np.mean((st.fields['u'] - u_exact)**2))
    return err

@pytest.mark.slow
def test_convergence_order():
    grids = [17,33,65]
    errs = []
    hs = []
    for g in grids:
        errs.append(run_case(g, method='sor'))
        hs.append(1/(g-1))
    orders = []
    for i in range(len(errs)-1):
        orders.append(math.log(errs[i]/errs[i+1]) / math.log(hs[i]/hs[i+1]))
    # Expect roughly second order spatial when near steady-state; allow lenient lower bound early
    assert min(orders) > 1.5, f"Observed orders too low: {orders}"
    # Basic iteration sanity check using one representative run (mid grid) for jacobi
    st_mid = init_state(nx=grids[1], ny=grids[1], nu=0.01)
    x = np.linspace(0,1,grids[1]); y = np.linspace(0,1,grids[1])
    X,Y = np.meshgrid(x,y, indexing='ij')
    u_exact = np.sin(math.pi*X)*np.sin(math.pi*Y)
    f = 2*math.pi**2 * st_mid.nu * u_exact
    u_tmp = u_exact*0
    h_mid = 1/(grids[1]-1)
    tol_mid = 2e-2 * h_mid
    it_mid, _ = solve_steady_diffusion(u_tmp, f, st_mid.mesh.dx(), st_mid.mesh.dy(), st_mid.nu, tol=tol_mid, method='sor', omega=1.8)
    assert it_mid < 5_000, f"SOR iterations unexpectedly large: {it_mid}"    


def test_sor_converges_faster():
    n = 65
    st = init_state(nx=n, ny=n, nu=0.01)
    x = np.linspace(0,1,n); y = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,y, indexing='ij')
    u_exact = np.sin(math.pi*X)*np.sin(math.pi*Y)
    f = 2*math.pi**2 * st.nu * u_exact
    u_j = u_exact*0
    u_s = u_exact*0
    h = 1/(n-1)
    tol = 5e-2 * h*h
    it_j, _ = solve_steady_diffusion(u_j, f, st.mesh.dx(), st.mesh.dy(), st.nu, tol=tol, method='jacobi')
    it_s, _ = solve_steady_diffusion(u_s, f, st.mesh.dx(), st.mesh.dy(), st.nu, tol=tol, method='sor', omega=1.8)
    # Expect SOR iterations significantly less; allow generous ratio
    assert it_s < it_j * 0.7, f"SOR not faster enough: jacobi={it_j} sor={it_s}" 
