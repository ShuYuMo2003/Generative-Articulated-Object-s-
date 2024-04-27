import ctypes
import os
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})

pyfusion_dir = os.path.dirname(os.path.realpath(__file__))
ctypes.cdll.LoadLibrary(os.path.join(pyfusion_dir, 'build', 'libfusion_cpu.so'))

from .cyfusion import *

# /home/shuyumo/research/GAO/external/mesh-fusion/libfusioncpu