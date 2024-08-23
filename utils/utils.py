import json
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .generate_obj_pic import generate_obj_pics


class HighPrecisionJsonEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, float):
            return format(obj, '.40f')
        return json.JSONEncoder.encode(self, obj)

def generate_special_tokens(dim, seed):
    np.random.seed(seed)
    token = np.random.normal(0, 1, dim).tolist()
    return token


def tokenize_part_info(part_info):
    token = []

    bounding_box = part_info['bounding_box']
    token += bounding_box[0] + bounding_box[1]

    joint_data_origin = part_info['joint_data_origin']
    token += joint_data_origin

    joint_data_direction = part_info['joint_data_direction']
    token += joint_data_direction

    limit = part_info['limit']
    token += limit

    latent_code = part_info['latent_code']
    token += latent_code

    return token

def untokenize_part_info(token):
    part_info = {
        'bounding_box': [token[0:3], token[3:6]],
        'joint_data_origin': token[6:9],
        'joint_data_direction': token[9:12],
        'limit': token[12:16],
        'latent_code': token[16:],
    }
    assert len(token[16:]) == 256
    return part_info

def generate_gif_toy(tokens, shape_output_path: Path, bar_prompt:str, n_frame: int=100,
                    n_timepoint: int=50, fps:int=40):

    def speed_control_curve(n_frame, n_timepoint, timepoint):
        frameid = n_frame*(1.0/(1+np.exp(
                -0.19*(timepoint-n_timepoint/2)))-0.5)+(n_frame/2)
        if frameid < 0:
            frameid = 0
        if frameid >= n_frame:
            frameid = n_frame-1
        return frameid

    buffers = []
    for ratio in tqdm(np.linspace(0, 1, n_frame), desc=bar_prompt):
        buffer = generate_obj_pics(tokens, ratio,
                [(-2.754140492463937, 1.5002379461105038, -3.2869899119443935),
                (0.003917710336454772, -0.03513334565514559, -0.09039018811838642),
                (0.20104799744040056, 0.9393848382797771, 0.2777333763437194)])
        buffers.append(buffer)

    frames = []
    for timepoint in range(n_timepoint):
        buffer_id = speed_control_curve(n_frame, n_timepoint, timepoint)
        frames.append(buffers[int(buffer_id)])

    frames = frames + frames[::-1]

    imageio.mimsave(shape_output_path.as_posix(), frames, fps=fps)