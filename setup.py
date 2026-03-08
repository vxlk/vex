"""Setup script for the vex Python package.

The C++ extension (_vex_core) is built separately via CMake.
This setup.py handles only the pure-Python package installation
and dependency management.  Use ``./dev.sh install`` for the
full build + install workflow.
"""

from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="vex",
    version="0.1.0",
    description="High-performance video thumbnail extraction",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "examples": [
            "numpy",
            "Pillow",
        ],
        "test": [
            "pytest",
            "Pillow",
        ],
        "bench": [
            "matplotlib",
        ],
    },
    package_data={
        "vex": ["*.pyd", "*.so", "*.dll"],
    },
)
