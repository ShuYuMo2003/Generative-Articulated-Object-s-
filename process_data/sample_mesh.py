import os
import numpy as np
from multiprocessing import Pool
from rich.progress import Progress

import sys
sys.path.append('..')
from utils.libmesh import check_mesh_contains


def export_points(mesh):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return


    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)

    points_surface = mesh.sample(n_points_surface)
    # print(points_surface)

    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies = check_mesh_contains(mesh, points)

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)


    print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale)
