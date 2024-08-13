from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from pathlib import Path
from transformer.utils import str2hash

class PartnetMobilityDataset(Dataset):
    def __init__(self, path:str, train:bool, sdf_dataset:bool, return_path_name=False):
        super().__init__()

        npzs_path = Path(path) / 'result'
        if not train:
            npzs_path = Path(path) / 'result' / 'test'

        self.files = list(npzs_path.glob('*.npz'))
        self.files.sort()
        self.return_path_name = return_path_name
        self.sdf_dataset = sdf_dataset

    def __len__(self):
        return len(self.files)

    def process_occ_dataset(self, point, sdf):
        enc_samplepoints, dec_samplepoints = point[0::2, ...], point[1::2, ...]
        enc_occ, dec_occ = (sdf[0::2, ...] < 0), (sdf[1::2, ...] < 0)

        # print('enc_samplepoints:', enc_samplepoints.shape)

        enc_occ = enc_occ.astype(np.float32)
        dec_occ = dec_occ.astype(np.float32)

        enc_samplepoints = enc_samplepoints.astype(np.float32)
        dec_samplepoints = dec_samplepoints.astype(np.float32)

        return_value = [
            torch.from_numpy(enc_samplepoints),
            torch.from_numpy(enc_occ),
            torch.from_numpy(dec_samplepoints),
            torch.from_numpy(dec_occ),
        ]

        return return_value

    def process_sdf_dataset(self, point, sdf):
        enc_samplepoints, dec_samplepoints = point[0::2, ...], point[1::2, ...]
        enc_sdf, dec_sdf = sdf[0::2, ...], sdf[1::2, ...]

        enc_sdf = enc_sdf.astype(np.float32)
        dec_sdf = dec_sdf.astype(np.float32)

        enc_samplepoints = enc_samplepoints.astype(np.float32)
        dec_samplepoints = dec_samplepoints.astype(np.float32)

        return_value = [
            torch.from_numpy(enc_samplepoints),
            torch.from_numpy(enc_sdf),
            torch.from_numpy(dec_samplepoints),
            torch.from_numpy(dec_sdf),
        ]

        return return_value

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file = np.load(file_path)

        point = file['point']
        sdf = file['sdf']

        indices = np.arange(point.shape[0])
        np.random.shuffle(indices)
        point = point[indices]
        sdf = sdf[indices]

        if not self.sdf_dataset:
            return_value = self.process_occ_dataset(point, sdf)
        else:
            return_value = self.process_sdf_dataset(point, sdf)

        if self.return_path_name:
            return_value.append(Path(self.files[idx]).stem)

        return return_value