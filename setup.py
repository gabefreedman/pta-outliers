#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:47:22 2020

@author: marvin
"""


import os
import sys
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Cython extensions
ext_modules=[
    Extension('jitterext',
             ['jitterext.pyx'],
             include_dirs = [numpy.get_include()],
             extra_compile_args=["-O2"]),
    Extension('choleskyext_omp',
             ['choleskyext_omp.pyx'],
             include_dirs = [numpy.get_include()],
             extra_link_args=["-liomp5"],
             extra_compile_args=["-O2", "-fopenmp", "-fno-wrapv"])
]


setup(
    ext_modules = cythonize(ext_modules)
)