import ctypes
import os

import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

#pyrender_dir = os.path.dirname(os.path.realpath(__file__))
#ctypes.cdll.LoadLibrary(os.path.join(pyrender_dir, 'pyrender.so'))
from .pyrender import *
