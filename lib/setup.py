from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="gtalign", 
    ext_modules=cythonize("gtalign.pyx"),
    include_dirs=[numpy.get_include()]
) 
