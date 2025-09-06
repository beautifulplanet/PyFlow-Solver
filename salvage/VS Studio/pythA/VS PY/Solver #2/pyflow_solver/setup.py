from setuptools import setup, Extension
import pybind11
import os
import numpy as np

# Define the C++ extension modules
ext_modules = [
    # Simple example module
    Extension(
        'pyflow_core',  # The name of the module in Python
        ['cpp/pyflow_core.cpp'],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),  # Include NumPy headers
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if os.name == 'nt' else ['-std=c++17'],
    ),
    
    # CFD solver module
    Extension(
        'pyflow_core_cfd',  # The name of the module in Python
        ['cpp/pyflow_core_cfd.cpp'],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),  # Include NumPy headers
            # Add Eigen's include path here if you use it
            # 'cpp/vendor/eigen',
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if os.name == 'nt' else ['-std=c++17'],
    ),
]

from setuptools import find_packages

setup(
    name='pyflow_hybrid',
    version='1.0',
    packages=['pyflow'],
    ext_modules=ext_modules,
    install_requires=['numpy', 'pybind11'],
    python_requires='>=3.7',
)
