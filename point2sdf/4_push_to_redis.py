import redis
import pickle
from rich import print
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm


_r = redis.Redis(host='localhost', port=6379, db=0)

raw_shape_info_paths = glob('output/1_preprocessed_info/*')

for raw_shape_info_path in tqdm(raw_shape_info_paths):

    raw_shape_info = np.load(raw_shape_info_path, allow_pickle=True).item()

    for single_part in raw_shape_info['part']:
        mesh_name = Path(single_part['mesh_off']).stem
        single_part['latent'] = np.load(f'output/3_generation_latent_code/latent_{mesh_name}.npy')
        del single_part['mesh_off']

    _r.set(Path(raw_shape_info_path).stem, pickle.dumps(raw_shape_info))

_r.save()

