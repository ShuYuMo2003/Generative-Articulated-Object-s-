from torch.utils.data import Dataset
import numpy as np
import torch

class PartnetMobilityDataset(Dataset):
    def __init__(self, path:list[tuple[str, str]], unpackbits=True):
        super().__init__()
        self.datapath = path
        self.unpackbits = unpackbits

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

        return (
            torch.from_numpy(pointcloud),
            torch.from_numpy(samplepoints),
            torch.from_numpy(occ)
        )