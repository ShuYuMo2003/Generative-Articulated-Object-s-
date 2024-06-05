import redis
import pickle
from rich import print
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import os

def main(config):
    _r = redis.Redis(host=config['host'], port=config['port'], db=config['db'])
    _r.flushdb()

    raw_shape_info_paths = glob(os.path.join(config['raw_shape_info_paths'], '*'))

    dec_data_path = str(Path(config['output_dataset_path']) / f'point-1')

    print('total we have ', len(raw_shape_info_paths), ' shapes to push to redis.')

    for raw_shape_info_path in tqdm(raw_shape_info_paths):

        raw_shape_info = np.load(raw_shape_info_path, allow_pickle=True).item()

        for single_part in raw_shape_info['part']:
            mesh_name = Path(single_part['mesh_off']).stem
            single_part['latent'] = np.load(os.path.join(config['latent_code_output_path'], f'latent_{mesh_name}.npy'))

            dec_data = np.load(os.path.join(dec_data_path, f'{mesh_name}.npz'))
            dec_samplepoint = dec_data['points'].astype(np.float32)
            dec_occ = np.unpackbits(dec_data['occupancies'])[:dec_samplepoint.shape[0]]
            dec_occ = dec_occ.astype(np.float32)

            single_part['dec_samplepoint'] = dec_samplepoint
            single_part['dec_occ'] = dec_occ
            del single_part['mesh_off']

        _r.set(Path(raw_shape_info_path).stem, pickle.dumps(raw_shape_info))

    with open(str(Path(config['latent_code_output_path']) / f'latent_generator_encode.txt'), 'r') as f:
        _r.set('latent_generator_encoder', f.read())

    _r.save()