import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "den",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['den.pyx'])
	)
