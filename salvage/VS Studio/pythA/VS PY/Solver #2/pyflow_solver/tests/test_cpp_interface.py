"""
Test to verify that C++ functions can modify NumPy arrays in-place.
This is a critical test for hybrid Python/C++ solvers.
"""

import numpy as np
import pytest
import sys
import os
# Try to import the C++ extension module
try:
    import pyflow_core_cfd
    CPP_TEST_INTERFACE_AVAILABLE = True
except ImportError:
    CPP_TEST_INTERFACE_AVAILABLE = False

@pytest.mark.skipif(not CPP_TEST_INTERFACE_AVAILABLE,
                    reason="C++ core module not available. Please compile it.")
def test_cpp_array_modification():
    """
    Test that C++ code can modify a NumPy array passed from Python.
    This is the fundamental verification for our hybrid solver.
    """
    # Ensure the function we want to test actually exists in the C++ module.
    # This prevents the test from passing silently if the function is missing.
    assert hasattr(pyflow_core_cfd, 'set_interior_values'), \
        "The required 'set_interior_values' function is not in the C++ module."

    # 1. Setup
    # Create a test array filled with zeros
    test_array = np.zeros((5, 5), dtype=np.float64)
    test_value = 42.0

    # 2. Action
    # Call the C++ function to modify the array in-place
    pyflow_core_cfd.set_interior_values(test_array, test_value)

    # 3. Assert
    # Create the array we expect after the C++ call
    expected_array = np.zeros((5, 5), dtype=np.float64)
    expected_array[1:-1, 1:-1] = test_value

    # Verify that the entire array matches the expected result.
    # This is a more robust check than verifying boundaries and interior separately.
    np.testing.assert_array_equal(test_array, expected_array,
                                  err_msg="C++ function did not modify the array as expected.")

if __name__ == "__main__":
    # A simple way to run the test directly for debugging.
    if CPP_TEST_INTERFACE_AVAILABLE:
        pytest.main([__file__])
    else:
        print("C++ test interface not available. Please compile it first.")
