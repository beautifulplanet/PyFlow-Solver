from setuptools import setup, find_packages

setup(
    name="pyfoamclone",
    version="0.1.0",
    packages=find_packages(),
    author="Your Name",
    description="A Python-based synthetic CFD scaffold (non-physical placeholder).",
    long_description=open('README.md').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    install_requires=[
        "numpy",
        "scipy"
    ],
    extras_require={
        'dev': ["pytest", "matplotlib", "ruff", "mypy", "pytest-cov", "coverage", "jsonschema", "radon"],
        'plot': ["matplotlib"],
        'test': ["pytest", "pytest-cov"],
        'cov': ["pytest-cov", "coverage"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        'console_scripts': [
            # 'pyfoamclone=pyfoamclone.main:main',
        ],
    },
)
