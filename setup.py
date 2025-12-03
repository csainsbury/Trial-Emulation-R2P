"""
Setup script for trial_emulation package.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Pybind11Extension(
        "trial_emulation._censor_func",
        [
            "cpp/censor_func.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

