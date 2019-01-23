from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
        ext_modules = cythonize("/home/robinson/Documents/visTools/visTools_v2/core_functions/kernels/gaussian_filter.pyx"),
        include_dirs=[np.get_include()],
        )
