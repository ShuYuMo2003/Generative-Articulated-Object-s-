from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
from pathlib import Path
from transformer.utils import str2hash

class PartnetMobilityDataset(Dataset):
    def __init__(self, path:str, train_ratio:float,
                 train:bool, selected_categories:list[str], return_path_name=False):
        super().__init__()
        assert 0 <= train_ratio <= 1
        selector = ((lambda x : str2hash(x) % 100 <  train_ratio * 100) if train
              else (lambda x : str2hash(x) % 100 >= train_ratio * 100))

        self.files = list(glob(str(Path(path) / f'result' / '*.npz')))
        # print(str(Path(path) / f'result' / '*.npz'))
        # print('all files:', list(map(lambda x : x.split('/')[-1].split('-')[0], self.files[:10])))
        self.files = list(filter(lambda x : x.split('/')[-1].split('_')[0] in selected_categories, self.files))
        if selected_categories != '*':
            self.files = list(filter(lambda x : x.split('/')[-1].split('_')[0] in selected_categories, self.files))
        # print('selected categories:', selected_categories)
        # print('selected files:', len(self.files))
        self.files.sort()
        # print('train: ', train)
        # print('files: ', self.files[:20])
        # print('files selected: ', list(map(selector, self.files[:20])))
        self.files = list(filter(selector, self.files))
        self.return_path_name = return_path_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        file = np.load(file_path)

        point = file['point']
        sdf = file['sdf']

        indices = np.arange(point.shape[0])
        np.random.shuffle(indices)
        point = point[indices]
        sdf = sdf[indices]

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
        if self.return_path_name:
            return_value.append(Path(self.files[idx]).stem)

        return return_value