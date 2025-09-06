import numpy as np
import os
import sys

# Add the current directory to path to find the compiled module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pyflow_core
    print("Successfully imported the C++ extension!")
except ImportError as e:
    print(f"Error importing the C++ extension: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    sys.exit(1)

# Create test arrays
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])

# Call the C++ function
result = pyflow_core.add_arrays(a, b)

print("--- Python Script ---")
print("Input A:\n", a)
print("Input B:\n", b)
print("Result from C++ core:\n", result)
print("\nIt works! The addition was performed at C++ speed.")
