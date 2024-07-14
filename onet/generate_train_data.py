import pymeshlab
from pathlib import Path
from rich.progress import track
from multiprocessing import cpu_count
import shutil
import os
import glob
import time
import math

MESH_FUSION_PATH = None
USE_GPU = True
PROCESS_COUNT = cpu_count() - 1

SCALE_STEP = 1
DEPTH_STEP = 2
WATER_STEP = 4
SAMPL_STEP = 8

BATCH_SIZE = 512


def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def process_batch(batch_files, objs_dir, out_dir, MESH_FUSION_PATH, step_mask):
    transform_dir = out_dir / 'trans'

    offs_dir = out_dir / 'offs'
    create_folder(offs_dir)
    scaled_dir = out_dir / 'scaled'
    create_folder(scaled_dir)
    depth_dir = out_dir / 'depth'
    create_folder(depth_dir)
    watertight_dir = out_dir / 'watertight'
    create_folder(watertight_dir)

    for repeat in range(2):
        create_folder(
            out_dir / f'pointcloud-{repeat}',
            out_dir / f'point-{repeat}',
            out_dir / f'watertight_scaled-{repeat}'
        )

    # Step 1: Convert to .off
    for stemname in batch_files:
        offs_file = offs_dir / (stemname + '.off')
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(objs_dir / (stemname + '.ply')))
        ms.save_current_mesh(str(offs_file))

    # Step 2: Scale
    if step_mask & SCALE_STEP:
        print(f'=== Scaling meshes...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '1_scale.py'}   \\
                            --n_proc {PROCESS_COUNT}                   \\
                            --in_dir {offs_dir}                        \\
                            --out_dir {scaled_dir}                     \\
                            --t_dir {transform_dir} ''') == 0
        print('done')

    # Step 3: Depth
    if step_mask & DEPTH_STEP:
        print(f'=== Creating depth maps...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                            --mode=render --n_proc {PROCESS_COUNT}     \\
                            --in_dir {scaled_dir}                      \\
                            --out_dir {depth_dir}''') == 0
        print('done')
    shutil.rmtree(scaled_dir)

    # Step 4: Watertight
    if step_mask & WATER_STEP:
        print(f'=== Producing watertight meshes...')
        assert os.system(f'''export use_gpu={int(USE_GPU)};            \\
                            python {MESH_FUSION_PATH / '2_fusion.py'}  \\
                            --mode=fuse --n_proc {PROCESS_COUNT}       \\
                            --in_dir {depth_dir}                       \\
                            --out_dir {watertight_dir}                 \\
                            --t_dir {transform_dir}''') == 0
        print('done')
    shutil.rmtree(depth_dir)
    shutil.rmtree(transform_dir)

    # Step 5: Sampling
    for repeat in range(2):
        pointcloud_dir = out_dir / f'pointcloud-{repeat}'
        point_dir = out_dir / f'point-{repeat}'
        watertight_scaled_dir = out_dir / f'watertight_scaled-{repeat}'
        if step_mask & SAMPL_STEP:
            current_file_path = os.path.realpath(__file__)
            current_file_path = os.path.dirname(current_file_path)
            print(f'=== Sampling from watertight meshes... (repeat={repeat})')
            assert os.system(f'''export use_gpu={int(USE_GPU)};
                                python {current_file_path}/sample_mesh.py {watertight_dir}            \\
                                    --n_proc {PROCESS_COUNT} --resize               \\
                                    --bbox_in_folder {offs_dir}                     \\
                                    --pointcloud_folder {pointcloud_dir}            \\
                                    --points_folder {point_dir}                    \\
                                    --mesh_folder {watertight_scaled_dir}          \\
                                    --packbits --float16 --overwrite                \\
                                    --pointcloud_size 50000                         \\
                                    --points_size 50000''') == 0
            print('done')

    # Clean up intermediate files
    shutil.rmtree(offs_dir)
    shutil.rmtree(watertight_dir)
    shutil.rmtree(out_dir / 'watertight_scaled-0')
    shutil.rmtree(out_dir / 'watertight_scaled-1')
    # shutil.rmtree(out_dir / 'pointcloud-0')
    # shutil.rmtree(out_dir / 'pointcloud-1')


def process_obj(objs_dir, out_dir, MESH_FUSION_PATH, step_mask=((1 << 5) - 1)):
    objs_dir = Path(objs_dir)
    out_dir = Path(out_dir)
    create_folder(out_dir)

    total_files = [file.stem for file in objs_dir.iterdir() if file.suffix == '.ply']
    print(f'\033[31mTotal files: {len(total_files)}\033[0m')
    total_batches = math.ceil(len(total_files) / BATCH_SIZE)

    for i in range(total_batches):
        timer = time.time()
        print(f'\033[31mProcessing batch {i + 1}/{total_batches}...\033[0m')
        batch_files = total_files[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        process_batch(batch_files, objs_dir, out_dir, MESH_FUSION_PATH, step_mask)
        print(f'\033[31mBatch {i + 1}/{total_batches} done in {time.time() - timer:.2f}s\033[0m')


if __name__ == '__main__':
    os.system('cp -r output /mnt/d/Research/data/output')