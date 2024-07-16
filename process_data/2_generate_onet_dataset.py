
import os
import shutil
from pathlib import Path
from time import sleep
from multiprocessing import Pool, cpu_count

import trimesh
from tqdm import tqdm
import mesh_to_sdf
import numpy as np
# import open3d as o3d

manifold_path = Path('../third_party/ManifoldPlus/build/manifold')

def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def ply_to_obj(ply_file, obj_file):
    mesh = trimesh.load_mesh(
        open(ply_file.as_posix(), 'rb'),
        file_type='ply'
    )
    (min_bound, max_bound) = mesh.bounds
    center = (max_bound + min_bound) / 2
    mesh.vertices -= center

    # print('1#', mesh.bounds)

    scale = (max_bound - min_bound).max() / (1 - 0.001)
    mesh.vertices /= scale

    # print('2#', mesh.bounds)

    mesh.export(obj_file.as_posix(), file_type='obj')


def obj_to_wtobj(obj_file, wt_obj_file):
    ok = os.system(f'''{manifold_path.as_posix()}     \\
                    --input {obj_file.as_posix()}     \\
                    --output {wt_obj_file.as_posix()} \\
                    > {wt_obj_file.as_posix()}.log ''')
    if ok != 0: raise RuntimeError(f'Error in watertight obj conversion: {obj_file}')

def wtobj_to_sdf(wt_obj_file, sdf_file):
    wt_obj = trimesh.load_mesh(
        open(wt_obj_file.as_posix(), 'r'),
        file_type='obj'
    )
    common_args = {
        'surface_point_method' : 'sample',
        'sign_method': 'normal',
    }
    # print('sampling sdf 1 1 1')
    # print('3#', wt_obj.bounds)

    # point, sdf = mesh_to_sdf.sample_sdf_near_surface(wt_obj, number_of_points=1000, **common_args)

    # in_range_indices = np.all((point >= -0.5) & (point <= 0.5), axis=1)
    # point = point[in_range_indices]
    # sdf = sdf[in_range_indices]

    # print("1 max = ", point.max(axis=0))
    # print("1 min = ", point.min(axis=0))

    box_size = 1.05
    uni_point = box_size * np.random.rand(110000, 3) - (box_size / 2)

    # print("2 max = ", uni_point.max(axis=0))
    # print("2 min = ", uni_point.min(axis=0))
    # print('sampling sdf 2 2 2')

    uni_sdf = mesh_to_sdf.mesh_to_sdf(wt_obj, uni_point, **common_args)
    # print('done done')

    point = uni_point # np.concatenate([point, uni_point], axis=0)
    sdf = uni_sdf # np.concatenate([sdf, uni_sdf], axis=0)

    min_bounds, max_bounds = wt_obj.bounds[0], wt_obj.bounds[1]
    outside_bounds = np.logical_or(np.any(point < min_bounds, axis=1), np.any(point > max_bounds, axis=1))
    sdf[outside_bounds] = 1

    print('saved', sdf_file)
    np.savez(sdf_file.as_posix(), point=point, sdf=sdf)
    return "Done"

    # selected_point = point[sdf < 0, ...]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(selected_point)
    # o3d.io.write_point_cloud(sdf_file.as_posix() + '0.ply', pcd)

    # selected_point = point[sdf > 0, ...]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(selected_point)
    # o3d.io.write_point_cloud(sdf_file.as_posix() + '1.ply', pcd)


def convert_mesh(ply_file, clear_temp=True):
    stem = ply_file.stem
    temp_dir = Path(f'../dataset/2_onet_dataset/temp/{stem}')
    result_dir = Path(f'../dataset/2_onet_dataset/result')

    create_folder(temp_dir, result_dir)

    obj_file = temp_dir / (stem + ".obj")
    wt_obj_file = temp_dir / (stem + ".wt.obj")
    sdf_target_file = result_dir / f'{stem}.sdf'

    print('Converting to (obj)', obj_file)
    ply_to_obj(ply_file, obj_file)
    print('Converting to (wt)', wt_obj_file)
    obj_to_wtobj(obj_file, wt_obj_file)
    print('Converting to (sdf)', sdf_target_file)
    wtobj_to_sdf(wt_obj_file, sdf_target_file)
    print('finished')

    if clear_temp: shutil.rmtree(temp_dir, ignore_errors=True)

    return "Done"

if __name__ == '__main__':
    shutil.rmtree('../dataset/2_onet_dataset', ignore_errors=True)
    all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply',
                            Path('../dataset/1_preprocessed_mesh/').iterdir()))
    # all_ply_files = [Path('../dataset/1_preprocessed_mesh/USB_1948_1.ply')]

    # convert_mesh(all_ply_files[0], False)
    # exit(0)

    with Pool(cpu_count() // 2) as p:
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

