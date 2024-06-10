from onet.dataset import PartnetMobilityDataset
from pathlib import Path
from torch.utils.data import DataLoader
from glob import glob
import torch
from onet.encoder_latent import Encoder
import numpy as np
import os
from rich import print

def main(config):

    if config['check_point_filename'] is None:
        ckpt_list = list(glob(os.path.join(config['check_point_path'], '*-*-encoder.ckpt')))
        ckpt_list.sort(key=lambda x : -float(x.split('-')[1]))
        check_point_path = ckpt_list[0]
    else:
        check_point_path = os.path.join(config['check_point_path'], config['check_point_filename'])

    print('selected ', check_point_path, ' as gen model')

    latent_code_output_path = config['latent_code_output_path']
    Path(latent_code_output_path).mkdir(exist_ok=True)

    dataset         = PartnetMobilityDataset(config['output_dataset_path'], train_ratio=1, train=True, return_path_name=True)
    dataloader      = DataLoader(dataset, batch_size=1, shuffle=False)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = torch.load(check_point_path, map_location=torch.device(device) ).to(device)

    with torch.no_grad():
        for idx, (enc_sp, enc_occ, dec_sp, dec_occ, path) in enumerate(dataloader):
            enc_sp = enc_sp.to(device)
            enc_occ = enc_occ.to(device)
            dec_sp = dec_sp.to(device)
            dec_occ = dec_occ.to(device)

            path = path[0]

            mean_z, logstd_z = encoder(enc_sp, enc_occ)

            latent = mean_z[0, ...]

            latent_numpy = latent.cpu().numpy()

            print(latent_numpy.shape, str(Path(latent_code_output_path) / f'latent_{path}'))
            np.save(str(Path(latent_code_output_path) / f'latent_{path}'), latent_numpy)
        with open(str(Path(latent_code_output_path) / f'latent_generator_encode.txt'), 'w') as f:
            f.write(check_point_path)