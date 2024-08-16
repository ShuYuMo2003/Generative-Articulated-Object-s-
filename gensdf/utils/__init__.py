import trimesh
from pathlib import Path
import numpy as np
import pyvista as pv

def generate_mesh_screenshot(mesh: trimesh.Trimesh) -> np.ndarray:

    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(mesh)
    plotter.show()

    return plotter.screenshot()