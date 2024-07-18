

import sys
import os
import json
import shutil
import trimesh

from tqdm import tqdm
import imageio
import numpy as np
from glob import glob
from pathlib import Path

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.generate_obj_pic import generate_obj_pics

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

def speed_control_curve(n_frame, n_timepoint, timepoint):
    frameid = n_frame*(1.0/(1+np.exp(-0.19*(timepoint-n_timepoint/2)))-0.5)+(n_frame/2)
    if frameid < 0:
        frameid = 0
    if frameid >= n_frame:
        frameid = n_frame-1
    return frameid

def generate_gif_toy():
    output_path = Path('../dataset/5_toy_gif_screenshot')
    output_path.mkdir(exist_ok=True)
    obj_info_paths = glob('../dataset/1_preprocessed_info/*')
    mesh_factory_path = '../dataset/1_preprocessed_mesh'
    obj_infos = [
        json.load(open(path, 'r'))
        for path in obj_info_paths
    ]
    for obj_info in obj_infos:
        print("displaying meta = ", obj_info['meta'])
        for part in obj_info['part']:
            part['mesh'] = trimesh.load_mesh(
                open(mesh_factory_path + '/' + part["mesh"], 'rb'),
                file_type='ply'
            )

        n_frame = 100
        n_timepoint = 50
        buffers = []
        for ratio in tqdm(np.linspace(0, 1, n_frame)):
            buffer = generate_obj_pics(obj_info['part'], ratio, camera_positions[0])
            buffers.append(buffer)

        frames = []
        for timepoint in range(n_timepoint):
            buffer_id = speed_control_curve(n_frame, n_timepoint, timepoint)
            frames.append(buffers[int(buffer_id)])

        frames = frames + frames[::-1]

        shape_id = obj_info['meta']['shape_id']
        path_id = obj_info['meta']['shape_path'].split('/')[-1]
        category = obj_info['meta']['catecory']

        shape_output_path = output_path / f"{category}_{shape_id}_{path_id}.gif"

        imageio.mimsave(shape_output_path.as_posix(), frames, fps=40)

def generate_screenshot_for_description(obj_info, camera_positions, n_pose):
    buffers = []
    for camera_positon in tqdm(camera_positions, desc='camera_positions'):
        for ratio in np.random.rand(n_pose):
            buffer = generate_obj_pics(obj_info['part'], ratio, camera_positon)
            buffers.append(buffer)
    return buffers

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
    for obj_info in obj_infos:
        print("process meta = ", obj_info['meta'])
        for part in obj_info['part']:
            part['mesh'] = trimesh.load_mesh(
                open(mesh_factory_path + '/' + part["mesh"], 'rb'),
                file_type='ply'
            )

        buffers = generate_screenshot_for_description(obj_info, camera_positions, 3)

        shape_id = obj_info['meta']['shape_id']
        path_id = obj_info['meta']['shape_path'].split('/')[-1]
        category = obj_info['meta']['catecory']

        shape_output_path = output_path / f"{category}_{shape_id}_{path_id}"
        shape_output_path.mkdir(exist_ok=True)

        for idx, buffer in enumerate(buffers):
            imageio.imwrite(shape_output_path / f"{category}_{shape_id}_{path_id}_{idx}.png", buffer)



