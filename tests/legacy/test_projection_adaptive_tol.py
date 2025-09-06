import os
import numpy as np
import pytest
from cfd_solver.pyflow.core import Mesh, SolverState
from cfd_solver.pyflow.core import projection_step

def make_state(n=33):
    mesh = Mesh(nx=n, ny=n)
    st = SolverState(mesh=mesh, fields={}, nu=0.0, rho=1.0)
    st.require_field('u',(n,n))
    st.require_field('v',(n,n))
    st.require_field('p',(n,n))
    return st

@pytest.mark.unit
def test_adaptive_tolerance_scales():  # LEGACY (will be moved to tests/legacy)
    os.environ['PROJECTION_ENABLE'] = '1'
    os.environ['PROJECTION_LINSOLVER'] = 'jacobi'
    os.environ['PROJECTION_POISSON_BASE_TOL'] = '1e-7'
    os.environ['PROJECTION_ADAPT_REF'] = '1e-4'
    os.environ['PROJECTION_ADAPTIVE_TOL'] = 'True'  # Enable adaptive tolerance with boolean string
    
    base_tol = 1e-7

    try:
        # Test with a deterministic high divergence field
        st_hi = make_state()
        # Create a divergent field pattern
        n = st_hi.fields['u'].shape[0]
        for i in range(n):
            for j in range(n):
                # Create velocity field with strong divergence
                st_hi.fields['u'][i,j] = 10.0 * (i/n - 0.5)  # -5 to 5 across the domain
                st_hi.fields['v'][i,j] = 10.0 * (j/n - 0.5)  # -5 to 5 across the domain
        
        stats_hi = projection_step(st_hi, dt=1e-4, use_advection=False, adaptive_dt=False)
        # Print available keys in notes to help debug
        print(f"Available keys in stats_hi.notes: {list(stats_hi.notes.keys())}")
        # Try alternate key names that might be used for the tolerance
        tol_hi = stats_hi.notes.get('poisson_tol', None) or stats_hi.notes.get('poisson_tolerance', None) or stats_hi.notes.get('adaptive_tol', None)
        assert tol_hi is not None, "poisson_tol not found in stats_hi.notes"
        assert tol_hi > base_tol, f"Adaptive tol should increase for large divergence, got {tol_hi}"
        
        # Small divergence should retain base tol
        st_lo = make_state()
        # Use uniform tiny values
        st_lo.fields['u'].fill(1e-10)
        st_lo.fields['v'].fill(-1e-10)
        stats_lo = projection_step(st_lo, dt=1e-4, use_advection=False, adaptive_dt=False)
        tol_lo = stats_lo.notes.get('poisson_tol', None)
        assert tol_lo is not None, "poisson_tol not found in stats_lo.notes"
        # For small divergence, tolerance should be close to base tolerance
        assert tol_lo <= 1.1 * base_tol, f"Tol should remain close to base for small divergence, got {tol_lo}"
    finally:
        # Cleanup environment variables
        for k in [
            'PROJECTION_ENABLE',
            'PROJECTION_LINSOLVER',
            'PROJECTION_POISSON_BASE_TOL',
            'PROJECTION_ADAPT_REF',
            'PROJECTION_ADAPTIVE_TOL'
        ]:
            os.environ.pop(k, None)
