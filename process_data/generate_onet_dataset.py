
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

    scale = (max_bound - min_bound).max() / (1 - 0.01)
    mesh.vertices /= scale
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
    point, sdf = mesh_to_sdf.sample_sdf_near_surface(wt_obj, number_of_points=50000, **common_args)

    boxsize = 2.1
    uni_point = boxsize * np.random.rand(10000, 3) - (boxsize / 2)
    uni_sdf = mesh_to_sdf.mesh_to_sdf(wt_obj, uni_point, **common_args)

    point = np.concatenate([point, uni_point], axis=0)
    sdf = np.concatenate([sdf, uni_sdf], axis=0)

    np.savez(sdf_file.as_posix(), point=point, sdf=sdf)

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

    print('Converting to (obj)', ply_file)
    ply_to_obj(ply_file, obj_file)
    print('Converting to (wt)', wt_obj_file)
    obj_to_wtobj(obj_file, wt_obj_file)
    print('Converting to (sdf)', sdf_target_file)
    wtobj_to_sdf(wt_obj_file, sdf_target_file)

    if clear_temp: shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    shutil.rmtree('../dataset/2_onet_dataset', ignore_errors=True)
    all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply',
                            Path('../dataset/1_preprecessed_mesh').iterdir()))


    with Pool(cpu_count() - 1) as p:
        result = [
            p.apply_async(convert_mesh, (ply_file, False))
            for ply_file in all_ply_files
        ]
        bar = tqdm(total=len(result), desc='Converting meshes')
        while result:
            for r in result:
                if r.ready():
                    result.remove(r)
                    bar.update(1)
            sleep(0.1)

