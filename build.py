import os
# See if Cython is installed

try:
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    print(" ------- WARNING: Cython Extensions not build sucessfully -------- ")

    def build(setup_kwargs):
        pass
# Cython is installed. Compile
else:
    print("  .. building C extensions")
    from setuptools import Extension
    from setuptools.dist import Distribution
    from distutils.command.build_ext import build_ext
    import numpy
    # This function will be executed in setup.py:
    def build(setup_kwargs):
        # The file you want to compile
        package = Extension('evomap.mapping.evomap._utils', ['src/evomap/mapping/evomap/_utils.pyx'], include_dirs=[numpy.get_include()])
        extensions = [package]

        # gcc arguments hack: enable optimizations
        os.environ['CFLAGS'] = '-O3'

        # Build
        setup_kwargs.update({
            'ext_modules': cythonize(
                extensions,
                language_level=3,
                compiler_directives={'linetrace': True},
            ),
            'cmdclass': {'build_ext': build_ext}
        })