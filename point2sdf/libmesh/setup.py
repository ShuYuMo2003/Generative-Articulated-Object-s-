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
      library_dirs=['./build/'],
      include_dirs=[np.get_include()]
    )
  ])
)
