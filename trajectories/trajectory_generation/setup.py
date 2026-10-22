"""Windows-friendly build for the trajectory_cpp pybind11 extension.

This replaces the original CMake-based setup.py, which is preserved as
``setup.py.orig`` and does not build on Windows/MSVC because:

  * ``CMakeLists.txt`` uses GCC-only flags (``-std=c++11``, ``-O3 -fPIC``);
  * it calls ``add_subdirectory(pybind11)`` but the ``pybind11`` git submodule
    was never checked out;
  * CMake >= 4 refuses ``cmake_minimum_required(VERSION 3.4...3.18)``;
  * on Windows a CMake MODULE library is written to the RUNTIME output dir, so
    the extension does not land where the original setup.py expects it.

This variant compiles the same three translation units directly through
setuptools + pybind11 and works on Windows (MSVC) and Linux alike.
"""

import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).parent.resolve()


def find_eigen_include() -> str:
    """Locate the Eigen 3 header root (the directory containing ``Eigen/Dense``)."""
    candidates = []

    env = os.environ.get("EIGEN_INCLUDE_DIR")
    if env:
        candidates.append(Path(env))

    # conda on Windows installs Eigen under <prefix>/Library/include/eigen3,
    # conda on Linux under <prefix>/include/eigen3.
    candidates += [
        Path(sys.prefix) / "Library" / "include" / "eigen3",
        Path(sys.prefix) / "include" / "eigen3",
        Path(sys.prefix) / "include",
        Path("/usr/include/eigen3"),
        Path("/usr/local/include/eigen3"),
        Path("/opt/homebrew/include/eigen3"),
    ]

    for candidate in candidates:
        if (candidate / "Eigen" / "Dense").is_file():
            return str(candidate)

    raise RuntimeError(
        "Eigen 3 headers were not found. Install them "
        "(e.g. `conda install -c conda-forge eigen=3.4.0`) or set the "
        "EIGEN_INCLUDE_DIR environment variable to the directory that "
        "contains 'Eigen/Dense'."
    )


ext_modules = [
    Pybind11Extension(
        "trajectory_cpp",
        sources=[
            "src/pywrap.cpp",
            "src/traj_gen.cpp",
            "src/trajectory_wrapper.cpp",
        ],
        include_dirs=[str(HERE / "include"), find_eigen_include()],
        define_macros=[
            # MSVC does not expose M_PI unless this is defined before <cmath> is
            # first parsed, and include/lib/root_finder.hpp uses M_PI.
            ("_USE_MATH_DEFINES", None),
        ],
        cxx_std=14,
    ),
]

setup(
    name="trajectory_cpp",
    version="0.0.1",
    author="Daniel Bin Schmid",
    author_email="danielbinschmid@outlook.de",
    description="A python wrapper for polynomial trajectory generation in C++.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)
