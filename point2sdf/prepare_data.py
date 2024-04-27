import pymeshlab
from pathlib import Path
from rich.progress import track
from multiprocessing import cpu_count
import shutil
import os
import glob

MESH_FUSION_PATH = Path('/home/shuyumo/research/GAO/external/mesh-fusion')


def create_folder(path):
    try: os.makedirs(str(path))
    except FileExistsError: pass

def process_obj(objs_dir, out_dir):
    cpu_cnt = 8 # cpu_count()
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
    print('=== Scaling meshes...')
    assert os.system(f'''python {MESH_FUSION_PATH / '1_scale.py'}  \\
                        --n_proc {cpu_cnt}                         \\
                        --in_dir {offs_dir}                        \\
                        --out_dir {scaled_dir}                     \\
                        --t_dir {transform_dir} ''') == 0
    print('done')

    depth_dir = out_dir / 'depth'
    create_folder(depth_dir)
    print('=== Create depths maps...')
    assert os.system(f'''python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                          --mode=render --n_proc {cpu_cnt}          \\
                          --in_dir {scaled_dir}                     \\
                          --out_dir {depth_dir}''') == 0
    print('done')

    watertight_dir = out_dir / 'watertight'
    create_folder(depth_dir)
    print('=== Produce watertight meshes...')
    assert os.system(f'''python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                        --mode=fuse --n_proc {cpu_cnt}              \\
                        --in_dir {depth_dir}                        \\
                        --out_dir {watertight_dir}                  \\
                        --t_dir {transform_dir}''') == 0
    print('done')





if __name__ == '__main__':
    in_dir  = Path('/home/shuyumo/research/GAO/dataset/100013/textured_objs')
    out_dir = Path('/home/shuyumo/research/GAO/point2sdf/output')

    shutil.rmtree(str(out_dir))
    create_folder(out_dir)

    process_obj(in_dir, out_dir)

    os.system('cp -r output /mnt/d/Research/data/output')