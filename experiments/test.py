from glob import glob
import trimesh
import pyvista as pv
import numpy as np
import json

shape_paths = glob('../dataset/raw/8867/textured_objs/*')
jsons = json.loads(open('../dataset/raw/8867/mobility_v2.json').read())


plotter = pv.Plotter()

start_points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# plotter.add_arrows(start_points, vectors, line_width=0.0001)

for idx, obj in enumerate(shape_paths):
    if obj.endswith('.mtl'):
        continue
    plotter.add_mesh(pv.read(obj), color='grey')

for k in jsons:
    try:
        axis = k['jointData']['axis']
    except Exception as e:
        continue
    start_point = np.array([axis['origin']])
    direction = np.array([axis['direction']])
    print(start_point, direction)
    plotter.add_arrows(start_point, direction, line_width=0.0001)


plotter.show()


