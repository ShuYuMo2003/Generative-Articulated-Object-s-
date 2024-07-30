

import sys
import os
import json
import time
import shutil
import trimesh

from tqdm import tqdm
import numpy as np
from glob import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# def generate_screen_shot(obj_parts):
#     dfn_to_parts = {
#         part['dfn']: part
#         for part in obj_parts
#     }
#     pass

camera_positions = [
        [(-2.754140492463937, 1.5002379461105038, -3.2869899119443935),
        (0.003917710336454772, -0.03513334565514559, -0.09039018811838642),
        (0.20104799744040056, 0.9393848382797771, 0.2777333763437194)],

        [(3.487083128152961, 1.8127192062148014, 1.9810015800028038),
        (-0.04570716149497277, -0.06563260832821388, -0.06195879116203942),
        (-0.37480300238091124, 0.9080915656577206, -0.18679512249404312)],
    ]


def generate_screenshot_for_description(obj_info, camera_positions, n_pose):
    buffers = []
    for camera_positon in camera_positions:
        for ratio in np.random.rand(n_pose):
            buffer = generate_obj_pics(obj_info['part'], ratio, camera_positon)
            buffers.append(buffer)
    return buffers

def generate_screenshot_wapper(obj_info):
    # print("process meta = ", obj_info['meta'])
    for part in obj_info['part']:
        if isinstance(part['mesh'], str):
            part['mesh'] = trimesh.load_mesh(
                open(mesh_factory_path + '/' + part["mesh"], 'rb'),
                file_type='ply'
            )

    buffers = generate_screenshot_for_description(obj_info, camera_positions, 3)

    shape_id = obj_info['meta']['shape_id']
    category = obj_info['meta']['catecory']

    shape_output_path = output_path / f"{category}_{shape_id}"
    shape_output_path.mkdir(exist_ok=True)

    for idx, buffer in enumerate(buffers):
        opath = shape_output_path / f"{idx}.png"
        print('[Write] : ', opath)
        imageio.imwrite(opath, buffer)

if __name__ == '__main__':
    # generate_gif_toy()

    output_path = Path('../dataset/4_screenshot')
    shutil.rmtree(output_path, ignore_errors=True)

    output_path.mkdir(exist_ok=True)

    obj_info_paths = glob('../dataset/1_preprocessed_info/*')
    mesh_factory_path = '../dataset/1_preprocessed_mesh'
    obj_infos = [
        json.load(open(path, 'r'))
        for path in obj_info_paths
    ]

    with Pool(cpu_count() - 1) as pool:
        result = [pool.apply_async(generate_screenshot_wapper, (obj_info,))
                    for obj_info in obj_infos]

        bar = tqdm(total=len(result), desc='generate_screenshot_wapper')

        while len(result) > 0:
            for res in result:
                if res.ready():
                    bar.update(1)
                    res.get()
                    result.remove(res)
            time.sleep(0.3)







