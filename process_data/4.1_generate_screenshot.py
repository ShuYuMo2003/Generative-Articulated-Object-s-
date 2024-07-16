

import sys
import os
import json
import trimesh

from tqdm import tqdm
import imageio
import numpy as np
from glob import glob

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

def speed_control_curve(n_frame, n_timepoint, timepoint):
    frameid = n_frame*(1.0/(1+np.exp(-0.19*(timepoint-n_timepoint/2)))-0.5)+(n_frame/2)
    if frameid < 0:
        frameid = 0
    if frameid >= n_frame:
        frameid = n_frame-1
    return frameid

if __name__ == '__main__':
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
            buffer = generate_obj_pics(obj_info['part'], ratio)
            buffers.append(buffer)

        frames = []
        for timepoint in range(n_timepoint):
            buffer_id = speed_control_curve(n_frame, n_timepoint, timepoint)
            frames.append(buffers[int(buffer_id)])

        frames = frames + frames[::-1]

        imageio.mimsave('output.gif', frames, fps=40)
