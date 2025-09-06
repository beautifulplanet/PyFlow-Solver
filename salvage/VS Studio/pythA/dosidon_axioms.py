# dosidon_axioms.py

import numpy as np

# Axiomatically Derived Core Constants & Parameters
# These values are not empirical inputs; they are derived from the foundational axioms.
# Reference: CPSAIL_Version_0_015_Prototype_.ipynb, 241 - Update dos 8-4 C.txt

# Fundamental Constants
EPSILON = -2.0  # Fundamental dimensionless coupling constant
N = 16          # Dimensionality of internal space
HBAR_PHYS = 2.0   # Physical Planck constant
C_PHYS = 1.0    # Speed of light in PsiPhi units
PI = np.pi      # Mathematical constant Pi

# Derived Parameters (numerical values are from project logs)
C_MU = 1502.87       # Derived combinatorial constant for viscosity
C_NU2 = 0.05         # Derived combinatorial constant for hyper-viscosity
GAMMA_F = 3.5        # Finitude Ansatz adiabatic index (3 + 1/2)

# These are derived from the core constants
L_PLANCK = np.sqrt(HBAR_PHYS / (C_PHYS**(1.5))) # Planck length
LAMBDA_D = L_PLANCK # A characteristic length scale for regularization
RHO_0 = 1.0         # Assumed equilibrium fluid density for this test case

# For use in solver
MU_FLUID = C_MU
NU2 = C_NU2

# Axiomatic Finitude Check
def check_finitude(value, name="value"):
    """
    Enforces Axiom 4: The Rejection of Zero and Infinity.
    Ensures that a value is finite, non-zero, and not NaN.
    """
    if not np.isfinite(value).all() or (value == 0.0).any():
        raise ValueError(f"Dosidon Axiomatic Violation: {name} is not finite or is zero.")
    return value
