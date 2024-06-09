from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from pathlib import Path
from transformer.utils import str2hash

class PartnetMobilityDataset(Dataset):
    def __init__(self, path:list[tuple[str, str]], train_ratio:float,
                 train:bool, unpackbits=True, return_path_name=False):
        super().__init__()
        assert 0 <= train_ratio <= 1
        selector = ( lambda x : str2hash(x[0]) % 100 <  train_ratio * 100 if train
                else lambda x : str2hash(x[0]) % 100 >= train_ratio * 100)
        enc_data = sorted(list(glob(str(Path(path) / f'point-0' / '*'))))
        dec_data = sorted(list(glob(str(Path(path) / f'point-1' / '*'))))
        assert len(dec_data) == len(enc_data)
        datasetpath = list(filter(selector, zip(enc_data, dec_data)))
        self.datapath = datasetpath
        self.unpackbits = unpackbits
        self.return_path_name = return_path_name

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        enc_samplepoints_dict = np.load(self.datapath[idx][0])
        dec_samplepoints_dict = np.load(self.datapath[idx][1])

        enc_samplepoints = enc_samplepoints_dict['points'].astype(np.float32)
        dec_samplepoints = dec_samplepoints_dict['points'].astype(np.float32)

        enc_occ = enc_samplepoints_dict['occupancies']
        dec_occ = dec_samplepoints_dict['occupancies']

        if self.unpackbits:
            enc_occ = np.unpackbits(enc_occ)[:enc_samplepoints.shape[0]]
            dec_occ = np.unpackbits(dec_occ)[:dec_samplepoints.shape[0]]

        enc_occ = enc_occ.astype(np.float32)
        dec_occ = dec_occ.astype(np.float32)

        return_value = [
            torch.from_numpy(enc_samplepoints),
            torch.from_numpy(enc_occ),
            torch.from_numpy(dec_samplepoints),
            torch.from_numpy(dec_occ),
        ]
        if self.return_path_name:
            return_value.append(Path(self.datapath[idx][0]).stem)

        return return_value