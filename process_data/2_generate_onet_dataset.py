
import os
import shutil
import time
from pathlib import Path
from time import sleep
from rich import print
from multiprocessing import Pool, cpu_count

import trimesh
from tqdm import tqdm
import numpy as np
# import open3d as o3d


manifold_path = Path('../third_party/ManifoldPlus/build/manifold')

total_number_of_sample_point = None
uniform_sample_ratio = None
surface_point_sigma = 0.02
points_padding = 0.05

def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def ply_to_obj(ply_file, obj_file):
    mesh = trimesh.load_mesh(
        open(ply_file.as_posix(), 'rb'),
        file_type='ply'
    )
    print('1# bounding box: ', mesh.bounds)
    (min_bound, max_bound) = mesh.bounds
    center = (max_bound + min_bound) / 2
    mesh.vertices -= center

    scale = (max_bound - min_bound).max() / (1 - 0.001)
    mesh.vertices /= scale

    print('2# bounding box: ', mesh.bounds)

    mesh.export(obj_file.as_posix(), file_type='obj')


def obj_to_wtobj_by_manifold(obj_file, wt_obj_file):
    ok = os.system(f'''{manifold_path.as_posix()}     \\
                    --input {obj_file.as_posix()}     \\
                    --output {wt_obj_file.as_posix()} \\
                    > {wt_obj_file.as_posix()}.log ''')
    if ok != 0: raise RuntimeError(f'Error in watertight obj conversion: {obj_file}')
    return "Done"

def obj_to_wtobj_by_pcu(obj_file, wt_obj_file):
    import point_cloud_utils as pcu

    v, f = obj_to_wtobj_by_pcu_vf(obj_file)
    pcu.save_mesh_vf(wt_obj_file.as_posix(), v, f)
    # print('done')
    return "Done"

def obj_to_wtobj_by_pcu_vf(obj_file):
    import point_cloud_utils as pcu

    # print('read obj file', obj_file)
    v, f = pcu.load_mesh_vf(obj_file)
    # print('done')


    # The resolution parameter controls the density of the output mesh
    # It is linearly proportional to the number of faces in the output
    # mesh. A higher value corresponds to a denser mesh.
    resolution = 19_000
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    # print('save watertight obj file', wt_obj_file)
    return vw, fw

# @see: https://www.fwilliams.info/point-cloud-utils/sections/mesh_sdf/
def wtobj_to_sdf_by_pcu(wt_obj_file, sdf_file, sdf_type):
    import point_cloud_utils as pcu
    _v, _f = pcu.load_mesh_vf(wt_obj_file.as_posix())

    _v = _v.astype(np.float64)

    n_point_total = total_number_of_sample_point * 2 # for encoder and decoder.
    n_point_near_surface = int(n_point_total * (1 - uniform_sample_ratio))

    # Generate point on the surface
    print('sampling point on mesh')
    fid, bc = pcu.sample_mesh_poisson_disk(_v, _f, num_samples=n_point_near_surface)
    point_near_surface = pcu.interpolate_barycentric_coords(_f, fid, bc, _v)
    n_point_near_surface = point_near_surface.shape[0]
    point_near_surface += surface_point_sigma * np.random.randn(n_point_near_surface, 3)
    print('point_near_surface : ', point_near_surface.shape)

    # Generate uniform point
    print('sampling point in box')
    n_point_uniform = n_point_total - n_point_near_surface
    box_size = 1 + points_padding
    uniform_point = box_size * np.random.rand(n_point_uniform, 3) - (box_size / 2)

    # Combine surface and uniform point
    query_pts = np.concatenate([point_near_surface, uniform_point], axis=0)
    # random shuffle
    np.random.shuffle(query_pts)

    print('query_pts: ', query_pts.shape)
    print('calculating sdf')
    sdf, fid, bc = pcu.signed_distance_to_mesh(query_pts, _v, _f)

    print('sdf: ', sdf)
    print('occ: ', np.sum(sdf < 0))
    print('occ rate: ', (sdf < 0).astype(np.float32).mean())

    print('saved', sdf_file)
    np.savez(sdf_file.as_posix(), point=query_pts, sdf=sdf)
    return "Done"

def wtobj_to_sdf_by_libmesh(wt_obj_file, sdf_file):
    import sys
    sys.path.append('..')
    from utils.libmesh import check_mesh_contains

    print(wt_obj_file.as_posix())
    wt_obj = trimesh.load_mesh(
        open(wt_obj_file.as_posix(), 'rb'),
        file_type='ply'
    )

    if not wt_obj.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % wt_obj_file)
        return

    n_point_total = total_number_of_sample_point * 2 # for encoder and decoder.
    n_point_near_surface = int(n_point_total * (1 - uniform_sample_ratio))

    point_near_surface = wt_obj.sample(n_point_near_surface)
    point_near_surface += surface_point_sigma * np.random.randn(n_point_near_surface, 3)

    n_point_uniform = n_point_total - n_point_near_surface
    box_size = 1 + points_padding
    uniform_point = box_size * np.random.rand(n_point_uniform, 3) - (box_size / 2)

    points = np.concatenate([point_near_surface, uniform_point], axis=0)

    print('points shape: ', points.shape)
    occupancies = check_mesh_contains(wt_obj, points)

    points = points.astype(np.float32)

    sdf = np.zeros_like(occupancies, dtype=np.float32)
    sdf[occupancies == True] = -1
    sdf[occupancies == False] = 1

    print('saved', sdf_file)
    np.savez(sdf_file.as_posix(), point=points, sdf=sdf)
    return "Done"

# @see: https://github.com/marian42/mesh_to_sdf   mei sha pi yong
def wtobj_to_sdf_by_mesh_to_sdf(wt_obj_file, sdf_file):
    import mesh_to_sdf
    wt_obj = trimesh.load_mesh(
        open(wt_obj_file.as_posix(), 'r'),
        file_type='obj'
    )
    common_args = {
        'surface_point_method' : 'sample',
        'sign_method': 'normal',
    }

    print('3# bounding box: ', wt_obj.bounds)

    n_point_total = total_number_of_sample_point * 2 # for encoder and decoder.
    n_point_near_surface = int(n_point_total * (1 - uniform_sample_ratio))
    n_point_uniform = n_point_total - n_point_near_surface

    # Generate point on the surface
    pointcloud = mesh_to_sdf.get_surface_point_cloud(wt_obj, **{'surface_point_method' : 'sample'})
    surface_point = pointcloud.get_random_surface_points(n_point_near_surface, use_scans=False)
    surface_point += surface_point_sigma * np.random.randn(n_point_near_surface, 3)

    # Generate uniform point
    box_size = 1 + points_padding
    uniform_point = box_size * np.random.rand(n_point_uniform, 3) - (box_size / 2)

    # Combine surface and uniform point
    point = np.concatenate([surface_point, uniform_point], axis=0)
    sdf = mesh_to_sdf.mesh_to_sdf(wt_obj, point, **common_args)

    min_bounds, max_bounds = wt_obj.bounds[0], wt_obj.bounds[1]
    outside_bounds = np.logical_or(np.any(point < min_bounds - 0.1, axis=1), np.any(point > max_bounds + 0.1, axis=1))
    sdf[outside_bounds] = 1

    print('point shape: ', point.shape)
    print('max: ', point.max(axis=0))
    print('min: ', point.min(axis=0))

    print('saved', sdf_file)
    np.savez(sdf_file.as_posix(), point=point, sdf=sdf)
    return "Done"

    # point, sdf = mesh_to_sdf.sample_sdf_near_surface(wt_obj, number_of_points=100000, **common_args)
    # # Centering and Scaling into [-0.5, 0.5]
    # # mesh_to_sdf: The mesh is first transformed to fit inside the unit sphere.
    # # @see: https://pypi.org/project/mesh-to-sdf/
    # valid_point = point[sdf < 0]
    # max_bounds = valid_point.max(axis=0)
    # min_bounds = valid_point.min(axis=0)
    # center = (max_bounds + min_bounds) / 2
    # point -= center

    # scale = (max_bounds - min_bounds).max() / (1 - 0.001)
    # point /= scale

    # # Remove points outside the [-0.5, 0.5] range
    # in_range_indices = np.all((point >= -0.51) & (point <= 0.51), axis=1)
    # point = point[in_range_indices]
    # sdf = sdf[in_range_indices]

    # n_points = len(point)

    # box_size = 1.005
    # uni_point = box_size * np.random.rand(200000, 3) - (box_size / 2)
    # uni_sdf = mesh_to_sdf.mesh_to_sdf(wt_obj, uni_point, **common_args)

    # point = np.concatenate([point, uni_point], axis=0)
    # sdf = np.concatenate([sdf, uni_sdf], axis=0)

def convert_mesh(ply_file, sdf_type, clear_temp, wt_method, sdf_method):
    start_time = time.time()
    stem = ply_file.stem
    temp_dir   = Path(f'../dataset/2_onet_{sdf_type}_dataset/temp/{stem}')
    result_dir = Path(f'../dataset/2_onet_{sdf_type}_dataset/result')

    assert sdf_type in ['sdf', 'occ'], f'Invalid sdf_type {sdf_type}, only support `sdf` or `occ`.'

    create_folder(temp_dir, result_dir)

    obj_file = temp_dir / (stem + ".obj")
    wt_obj_file = temp_dir / (stem + ".wt.ply")
    sdf_target_file = result_dir / f'{stem}.sdf'

    print('(1) Converting to (obj)', obj_file)
    ply_to_obj(ply_file, obj_file)

    print('(2) Converting to (wt)', wt_obj_file)
    if wt_method == 'pcu':
        print('(2) Using pcu to watertight obj')
        assert obj_to_wtobj_by_pcu(obj_file, wt_obj_file) == 'Done'
    elif wt_method == 'manifold':
        print('(2) Using manifold to watertight obj')
        assert obj_to_wtobj_by_manifold(obj_file, wt_obj_file) == 'Done'
    else:
        raise ValueError(f'Invalid method wt_method {wt_method}')

    print('(3) Converting to (sdf)', sdf_target_file)
    if sdf_method == 'pcu':
        print('(3) Using pcu to generate sdf')
        assert wtobj_to_sdf_by_pcu(wt_obj_file, sdf_target_file, sdf_type) == 'Done'
    elif sdf_method == 'mesh_to_sdf':
        print('(3) Using mesh_to_sdf to generate sdf')
        if sdf_type == 'sdf': raise ValueError('Not implemented sdf_type `sdf` for mesh_to_sdf')
        assert wtobj_to_sdf_by_mesh_to_sdf(wt_obj_file, sdf_target_file, sdf_type) == 'Done'
    elif sdf_method == 'libmesh':
        print('(3) Using libmesh to generate sdf')
        if sdf_type == 'sdf': raise ValueError('Not implemented sdf_type `sdf` for libmesh')
        if wtobj_to_sdf_by_libmesh(wt_obj_file, sdf_target_file) != 'Done':
            print('Error in sdf generation')
            return 'Error'
    else:
        raise ValueError(f'Invalid method sdf_method {sdf_method}')

    print('finished in ', time.time() - start_time, 's')

    if clear_temp: shutil.rmtree(temp_dir, ignore_errors=True)

    return "Done"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate SDF from mesh files.')
    parser.add_argument('--wt_method', type=str, default='pcu', choices=['pcu', 'manifold'], help='Method for watertight conversion')
    parser.add_argument('--sdf_method', type=str, default='pcu', choices=['pcu', 'mesh_to_sdf', 'libmesh'], help='Method for SDF generation')
    parser.add_argument('--sdf_type', type=str, default='occ', choices=['occ', 'sdf'], help='Type of SDF')
    parser.add_argument('--clear_temp_file', type=bool, default=False, help='Clear temp files')

    parser.add_argument('--n_point_each', type=int, required=True, help='Number of points for each mesh')
    parser.add_argument('--uniform_sample_ratio', type=float, required=True, help='Uniform sample ratio')
    parser.add_argument('--n_process', type=int, required=True, help='Number of process')

    args = parser.parse_args()

    total_number_of_sample_point = args.n_point_each
    uniform_sample_ratio = args.uniform_sample_ratio



    shutil.rmtree('../dataset/2_onet_{args.sdf_type}_dataset', ignore_errors=True)
    all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply',
                            Path('../dataset/1_preprocessed_mesh/').iterdir()))


    # all_ply_files = [Path('../dataset/1_preprocessed_mesh/USB_64_0.ply')]

    # convert_mesh(all_ply_files[0], args.sdf_type,
    #                                args.clear_temp_file, args.wt_method, args.sdf_method)
    # exit(0)

    failed = []
    with Pool(args.n_process) as p:
        result = [
            (p.apply_async(convert_mesh, (ply_file, args.sdf_type,
                                          args.clear_temp_file, args.wt_method, args.sdf_method)), ply_file)
            for ply_file in all_ply_files
        ]
        bar = tqdm(total=len(result), desc='Converting meshes')
        while result:
            for r, ply_file in result:
                if r.ready():
                    bar.update(1)
                    if r.get() != 'Done':
                        print('Error in converting mesh')
                        failed.append(ply_file)
                    result.remove((r, ply_file))
            sleep(0.1)

    print('Failed: ', failed)

