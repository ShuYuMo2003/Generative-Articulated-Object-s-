import random

import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from utils.logging import Log, console

class GenSDFDataset(Dataset):
    def __init__(self, dataset_dir: Path, train: bool,
                 samples_per_mesh: int, pc_size: int,
                 uniform_sample_ratio: float, cache_size: int):
        super().__init__()

        self.dataset_dir = list(dataset_dir.glob('result/*.npz' if train else 'result/test/*.npz'))
        random.shuffle(self.dataset_dir)

        self.n_uniform_point = int(samples_per_mesh * uniform_sample_ratio)
        self.n_near_surfcae_point = samples_per_mesh - self.n_uniform_point
        self.pc_size = pc_size

        self.data = [None] * len(self.dataset_dir)
        total_cache_size = min(len(self.dataset_dir), cache_size)
        Log.info(f'Loading {total_cache_size} data into memory')
        for i in tqdm(range(total_cache_size),
                      desc='Load Caching Data into Memory', total=total_cache_size):
            file = self.dataset_dir[i]
            data = np.load(file.as_posix(), allow_pickle=True)
            self.data[i] = data


    def __len__(self):
        return len(self.dataset_dir)

    def select_point(self, point, sdf, n_point):
        half = int(n_point / 2)

        neg_idx = np.where(sdf < 0)
        pos_idx = np.where(~(sdf < 0))

        assert len(neg_idx) == 1 and len(pos_idx) == 1

        neg_idx = neg_idx[0]
        pos_idx = pos_idx[0]

        assert neg_idx.shape[0] >= half or pos_idx.shape[0] >= half, 'Not enough points'

        np.random.shuffle(neg_idx)
        np.random.shuffle(pos_idx)

        if neg_idx.shape[0] < half:
            n_point_from_other = half - neg_idx.shape[0]
            neg_idx = np.concatenate((neg_idx, pos_idx[-n_point_from_other:]))
            pos_idx = pos_idx[:-n_point_from_other]

        if pos_idx.shape[0] < half:
            n_point_from_other = half - pos_idx.shape[0]
            pos_idx = np.concatenate((pos_idx, neg_idx[-n_point_from_other:]))
            neg_idx = neg_idx[:-n_point_from_other]

        assert neg_idx.shape[0] >= half and pos_idx.shape[0] >= half, 'Not enough points'

        neg_idx = neg_idx[:half]
        pos_idx = pos_idx[:half]

        idx = np.concatenate([neg_idx, pos_idx])
        np.random.shuffle(idx)

        return point[idx], sdf[idx]

    def __getitem__(self, index):
        file = self.dataset_dir[index]
        if self.data[index] is not None:
            data = self.data[index]
            # Log.info(f'Using cached data for index {index}')
        else:
            data = np.load(file.as_posix(), allow_pickle=True)

        uniform_point, uniform_sdf = self.select_point(data['point_uniform'], data['sdf_uniform'], self.n_uniform_point)
        surface_point, surface_sdf = self.select_point(data['point_surface'], data['sdf_surface'], self.n_near_surfcae_point)

        point_cloud = data['point_on']
        np.random.shuffle(point_cloud)
        point_cloud = point_cloud[:self.pc_size]

        # Convert to float32
        uniform_point   = uniform_point.astype(np.float32)
        uniform_sdf     = uniform_sdf.astype(np.float32)
        surface_point   = surface_point.astype(np.float32)
        surface_sdf     = surface_sdf.astype(np.float32)
        point_cloud     = point_cloud.astype(np.float32)

        xyz = np.concatenate([uniform_point, surface_point])
        gt_sdf = np.concatenate([uniform_sdf, surface_sdf])

        idx = np.random.permutation(xyz.shape[0])

        return {
            'xyz': xyz[idx],
            'gt_sdf': gt_sdf[idx],
            'point_cloud': point_cloud,
            'filename': file.stem
        }