import pymeshlab
from pathlib import Path
from rich.progress import track
from multiprocessing import cpu_count
import shutil
import os
import glob

MESH_FUSION_PATH = Path('/home/shuyumo/research/GAO/point2sdf/mesh-fusion')
USE_GPU = False
PROCESS_COUNT = 8 # cpu_count()

SCALE_STEP = 1
DEPTH_STEP = 2
WATER_STEP = 4
SAMPL_STEP = 8


def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def process_obj(objs_dir, out_dir, step_mask=(1<<5)-1):
    total_stemname = [file.stem for file in objs_dir.iterdir() if file.suffix == '.obj']
    transform_dir = out_dir / 'trans'

    offs_dir = out_dir / 'offs'
    create_folder(offs_dir)
    for stemname in track(total_stemname, description="Convert objs 2 offs"):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(objs_dir / (stemname + '.obj')))
        ms.save_current_mesh(str(out_dir / 'offs' / (stemname + '.off')))


    scaled_dir = out_dir / 'sacled'
    create_folder(scaled_dir)
    if step_mask & SCALE_STEP:
        print('=== Scaling meshes...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '1_scale.py'}   \\
                            --n_proc {PROCESS_COUNT}                   \\
                            --in_dir {offs_dir}                        \\
                            --out_dir {scaled_dir}                     \\
                            --t_dir {transform_dir} ''') == 0
        print('done')

    depth_dir = out_dir / 'depth'
    create_folder(depth_dir)
    if step_mask & DEPTH_STEP:
        print('=== Create depths maps...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                            --mode=render --n_proc {PROCESS_COUNT}     \\
                            --in_dir {scaled_dir}                      \\
                            --out_dir {depth_dir}''') == 0
        print('done')

    watertight_dir = out_dir / 'watertight'
    create_folder(depth_dir)
    if step_mask & WATER_STEP:
        print('=== Produce watertight meshes...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                            --mode=fuse --n_proc {PROCESS_COUNT}       \\
                            --in_dir {depth_dir}                       \\
                            --out_dir {watertight_dir}                 \\
                            --t_dir {transform_dir}''') == 0
        print('done')


    pointcloud_dir          = out_dir / 'pointcloud'
    point_dir               = out_dir / 'point'
    watertight_scaled_dir   = out_dir / 'watertight_scaled'
    create_folder(watertight_scaled_dir, pointcloud_dir, point_dir)
    if step_mask & SAMPL_STEP:
        print('=== Samplig from watertight meshes...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};
                            python sample_mesh.py {watertight_dir}              \\
                                --n_proc {PROCESS_COUNT} --resize               \\
                                --bbox_in_folder {offs_dir}                     \\
                                --pointcloud_folder {pointcloud_dir}            \\
                                --points_folder {point_dir}                     \\
                                --mesh_folder {watertight_scaled_dir}           \\
                                --packbits --float16 --overwrite''') == 0
        print('done')





if __name__ == '__main__':
    in_dir  = Path('/home/shuyumo/research/GAO/dataset/100013/textured_objs')
    out_dir = Path('/home/shuyumo/research/GAO/point2sdf/output')

    # shutil.rmtree(str(out_dir))
    # create_folder(out_dir)

    process_obj(in_dir, out_dir, SAMPL_STEP)

    os.system('cp -r output /mnt/d/Research/data/output')