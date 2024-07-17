
import os
import shutil
from pathlib import Path
from time import sleep
from multiprocessing import Pool, cpu_count


import trimesh
from tqdm import tqdm
import mesh_to_sdf
import point_cloud_utils as pcu
import numpy as np
# import open3d as o3d

manifold_path = Path('../third_party/ManifoldPlus/build/manifold')

total_number_of_sample_point = 50000
uniform_sample_ratio = 0.9
surface_point_sigma = 0.03
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


def obj_to_wtobj(obj_file, wt_obj_file):
    ok = os.system(f'''{manifold_path.as_posix()}     \\
                    --input {obj_file.as_posix()}     \\
                    --output {wt_obj_file.as_posix()} \\
                    > {wt_obj_file.as_posix()}.log ''')
    if ok != 0: raise RuntimeError(f'Error in watertight obj conversion: {obj_file}')

# @see: https://www.fwilliams.info/point-cloud-utils/sections/mesh_sdf/
def wtobj_to_sdf_by_pcu(wt_obj_file, sdf_file):
    _v, _f = pcu.load_mesh_vf(wt_obj_file)

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
    print('query_pts: ', query_pts.shape)
    print('calculating sdf')
    sdf, fid, bc = pcu.signed_distance_to_mesh(query_pts, _v, _f)

    print('sdf: ', sdf)
    print('occ: ', np.sum(sdf < 0))

    print('saved', sdf_file)
    np.savez(sdf_file.as_posix(), point=query_pts, sdf=sdf)
    return "Done"

# @see: https://github.com/marian42/mesh_to_sdf   mei sha pi yong
def wtobj_to_sdf_by_mesh_to_sdf(wt_obj_file, sdf_file):
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

def convert_mesh(ply_file, clear_temp=True, method='pcu'):
    stem = ply_file.stem
    temp_dir = Path(f'../dataset/2_onet_v2_dataset/temp/{stem}')
    result_dir = Path(f'../dataset/2_onet_v2_dataset/result')

    create_folder(temp_dir, result_dir)

    obj_file = temp_dir / (stem + ".obj")
    wt_obj_file = temp_dir / (stem + ".wt.obj")
    sdf_target_file = result_dir / f'{stem}.sdf'

    print('Converting to (obj)', obj_file)
    ply_to_obj(ply_file, obj_file)
    print('Converting to (wt)', wt_obj_file)
    obj_to_wtobj(obj_file, wt_obj_file)
    print('Converting to (sdf)', sdf_target_file)
    if method == 'pcu':
        print('Using pcu method')
        wtobj_to_sdf_by_pcu(wt_obj_file, sdf_target_file)
    else:
        print('Using mesh_to_sdf method')
        wtobj_to_sdf_by_mesh_to_sdf(wt_obj_file, sdf_target_file)
    print('finished')

    if clear_temp: shutil.rmtree(temp_dir, ignore_errors=True)

    return "Done"

if __name__ == '__main__':
    shutil.rmtree('../dataset/2_onet_v2_dataset', ignore_errors=True)
    all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply',
                            Path('../dataset/1_preprocessed_mesh/').iterdir()))

    # all_ply_files = [Path('../dataset/1_preprocessed_mesh/USB-100061-0.ply')]

    # convert_mesh(all_ply_files[0], False, 'pcu')
    # exit(0)

    with Pool(cpu_count() - 2) as p:
        result = [
            p.apply_async(convert_mesh, (ply_file, False))
            for ply_file in all_ply_files
        ]
        bar = tqdm(total=len(result), desc='Converting meshes')
        while result:
            for r in result:
                if r.ready():
                    bar.update(1)
                    assert r.get() == 'Done'
                    result.remove(r)
            sleep(0.1)

