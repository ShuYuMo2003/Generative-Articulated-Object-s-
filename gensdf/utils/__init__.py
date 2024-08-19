import trimesh
from pathlib import Path
import numpy as np
import pyvista as pv

def generate_mesh_screenshot(mesh: trimesh.Trimesh) -> np.ndarray:

    pv.global_theme.allow_empty_mesh = True

    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(mesh)
    plotter.show()

    return plotter.screenshot()