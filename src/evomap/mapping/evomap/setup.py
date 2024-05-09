"""
Build setup for C extensions.
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension('_utils', ['_utils.pyx'], 
              include_dirs=[numpy.get_include()])
    ]

setup(
    ext_modules=cythonize(ext_modules, language_level="3")
)
