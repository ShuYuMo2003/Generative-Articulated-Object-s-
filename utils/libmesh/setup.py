from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from torch.utils.cpp_extension import BuildExtension
from distutils.extension import Extension

setup(
  name="triangle_hash",
  cmdclass= {'build_ext': BuildExtension},
  ext_modules=cythonize([
    Extension('triangle_hash',
      sources=['triangle_hash.pyx'],
      libraries=['m'],  # Unix-like specific
      include_dirs=[np.get_include()]
    )
  ])
)
