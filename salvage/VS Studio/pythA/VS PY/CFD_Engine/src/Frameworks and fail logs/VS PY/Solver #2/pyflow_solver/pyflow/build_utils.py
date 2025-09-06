"""
A module for building C++ extension setup instructions.
This file helps with building the PyFlow C++ extension modules.
"""

import os
import sys
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# Determine whether we're on Windows or not
is_windows = os.name == 'nt'

# Helper function to find installed pybind11
def find_pybind11_path():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        print("Error: pybind11 is required for building C++ extensions.")
        print("Please install it with: pip install pybind11")
        sys.exit(1)

# Helper function to find NumPy include path
def find_numpy_path():
    try:
        import numpy as np
        return np.get_include()
    except ImportError:
        print("Error: NumPy is required for building C++ extensions.")
        print("Please install it with: pip install numpy")
        sys.exit(1)

# Define compiler flags based on platform
def get_compile_flags():
    flags = []
    if is_windows:
        flags = ['/std:c++17', '/O2', '/DNDEBUG']
        
        # Check for AVX2 support on Windows
        # In a real implementation, we might use a more robust check
        flags.append('/arch:AVX2')
    else:
        flags = ['-std=c++17', '-O3', '-DNDEBUG', '-march=native']
        
        # Add more flags for gcc/clang
        extra_flags = ['-fvisibility=hidden']
        for flag in extra_flags:
            flags.append(flag)
    
    return flags

# Define the extension modules
def get_extension_modules():
    # Get compiler flags
    compile_flags = get_compile_flags()
    
    # Define the extension modules
    extensions = [
        # Basic utilities module
        Extension(
            'pyflow_core',
            sources=['cpp/pyflow_core.cpp'],
            include_dirs=[
                find_pybind11_path(),
                find_numpy_path(),
            ],
            language='c++',
            extra_compile_args=compile_flags,
        ),
        
        # CFD solver module
        Extension(
            'pyflow_core_cfd',
            sources=['cpp/pyflow_core_cfd.cpp'],
            include_dirs=[
                find_pybind11_path(),
                find_numpy_path(),
                # Add Eigen's include path here if you use it
                # 'cpp/vendor/eigen',
            ],
            language='c++',
            extra_compile_args=compile_flags,
        ),
    ]
    
    return extensions

# Custom build extension class
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    
    def build_extensions(self):
        build_ext.build_extensions(self)
