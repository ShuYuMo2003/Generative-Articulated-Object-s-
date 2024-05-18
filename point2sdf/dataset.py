from torch.utils.data import Dataset
import numpy as np
import torch
import math
from pathlib import Path

class PartnetMobilityDataset(Dataset):
    def __init__(self, path:list[tuple[str, str]], train_ratio:float,
                 train:bool, unpackbits=True, return_path_name=False):
        super().__init__()
        train_count = int(len(path) * train_ratio)
        self.datapath = path[:train_count] if train else path[train_count:]
        self.unpackbits = unpackbits
        self.return_path_name = return_path_name

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):
        pointcloud_dict = np.load(self.datapath[idx][0])
        pointcloud = pointcloud_dict['points']
        pointcloud = pointcloud.astype(np.float32)

        samplepoints_dict = np.load(self.datapath[idx][1])
        samplepoints = samplepoints_dict['points']

        sampled_point_cnt = samplepoints.shape[0]

        # Copy from onet code.
        if samplepoints.dtype == np.float16:
            samplepoints = samplepoints.astype(np.float32)
            samplepoints += 1e-4 * np.random.randn(*samplepoints.shape)
        else:
            samplepoints = samplepoints.astype(np.float32)

        occ = samplepoints_dict['occupancies']
        if self.unpackbits:
            occ = np.unpackbits(occ)[:sampled_point_cnt]
        occ = occ.astype(np.float32)

        # print(pointcloud.shape)
        # print(samplepoints.shape)
        # print(occ.shape)

        return_value = [
            torch.from_numpy(pointcloud),
            torch.from_numpy(samplepoints),
            torch.from_numpy(occ)
        ]
        if self.return_path_name:
            return_value.append(Path(self.datapath[idx][0]).stem)

        return return_value