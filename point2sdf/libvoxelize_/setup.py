from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from Cython.Distutils import build_ext
from distutils.extension import Extension

setup(
  name="voxelize",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('voxelize',
      ['voxelize.pyx'],
      language='c++',
      library_dirs=['./build/'],
      include_dirs=[np.get_include()]
    )
  ]
)
