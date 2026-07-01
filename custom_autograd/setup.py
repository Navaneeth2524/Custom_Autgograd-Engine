from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "custom_matrix_ops",
        ["bindings.cpp", "matrix_ops.cpp"],
        # Optimize code execution and specify modern C++ standard
        extra_compile_args=["-O3", "-std=c++11", "-march=native"],
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="custom_matrix_ops",
    version=__version__,
    author="Antigravity",
    description="C++ matrix operation extensions for custom autograd engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
