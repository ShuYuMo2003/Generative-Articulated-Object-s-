from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
os.environ['CC'] = '/usr/bin/gcc'

setup(
  cmdclass= {'build_ext': build_ext},
  ext_modules=cythonize([
    Extension(
      'pykdtree.kdtree',
      sources=[
        'pykdtree/kdtree.c',
        'pykdtree/_kdtree_core.c'
      ],
      language='c',
      extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
      extra_link_args=['-lgomp'],
      include_dirs=[np.get_include()]
    )
  ])
)
