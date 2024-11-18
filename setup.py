from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "ifield.solvers",
        ["src/ifield/solvers.pyx"],
        extra_compile_args=["-O2"],
        include_dirs=[np.get_include()]
    ),
]

setup(
    ext_modules = cythonize(
        ext_modules,
        language_level="3",
    ),
)
