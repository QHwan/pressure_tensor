import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "pressure_ik",
	include_dirs = [np.get_include()],
	ext_modules=cythonize(['pressure_ik.pyx','func.pyx'])
	)
