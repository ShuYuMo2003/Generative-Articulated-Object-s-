from torch.utils.data import Dataset
import numpy as np
import redis
import pickle
import torch
from rich import print
from tqdm import tqdm
import copy

def identity_or_create_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class FromRedisNativeDataset(Dataset):
    def __init__(self, host, port, db, fix_length,
                    seed=(hash('love ytq.') & ((1 << 25) - 1))):
        super().__init__()
        self.fix_length = fix_length
        self._r = redis.Redis(host=host, port=port, db=db)

        def load_specific_shape(part_key):
            return pickle.loads(self._r.get(part_key))

        part_keys = tuple(map(lambda x : x.decode('utf-8'), self._r.keys()))
        self.info = [ (part_key, idx)
                        for part_key in tqdm(part_keys, desc='Checking data from redis')
                            for idx in range(len(load_specific_shape(part_key)['part']) + 1) ]

        self.basic_shape_structure = [
            ('dfn', 1), ('dfn_fa', 1), ('origin', 3), ('direction', 3),
            ('bounds', 6), ('tran', 3), ('limit', 4), ('latent', 128)
        ]

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        self.s_part = self.generate_random_part();  self.s_part['dfn'], self.s_part['dfn_fa'] = 0,          0
        self.e_part = self.generate_random_part();  self.e_part['dfn'], self.e_part['dfn_fa'] = fix_length, fix_length
        # placeholder, dfn and dfn_fa should be fixed same as e_part.
        self.p_part = self.generate_zero_part();    self.p_part['dfn'], self.p_part['dfn_fa'] = fix_length+1, fix_length
        self.g_part = self.generate_zero_part(); # `dfn`, `dfn_fa` should be same as generative part.


    def generate_random_part(self):
        return { key : torch.randn(dim) for key, dim in self.basic_shape_structure }

    def generate_zero_part(self):
        return { key : torch.zeros(dim) for key, dim in self.basic_shape_structure }

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        part_key, part_dataset_idx = self.info[index]
        bytedata = self._r.get(part_key)

        if not bytedata:
            raise RuntimeError(f"key {part_key} not found")
        data = pickle.loads(bytedata)
        if not isinstance(data, dict):
            raise RuntimeError(f"key {part_key} is not dict.")

        S = copy.deepcopy(self.s_part) # start part tag
        G = copy.deepcopy(self.g_part) # generative part tag
        P = copy.deepcopy(self.p_part) # placeholder part tag
        E = copy.deepcopy(self.e_part) # end part tag

        prediction_part = data['part'][part_dataset_idx] if part_dataset_idx < len(data['part']) else E


        dataset_for_train = [G, S]

        for i in range(part_dataset_idx):
            dataset_for_train.append(data['part'][i])

        G['dfn'], G['dfn_fa'] = dataset_for_train[-1]['dfn'], dataset_for_train[-1]['dfn_fa']

        for i in range(self.fix_length - len(dataset_for_train)):
            dataset_for_train.append(P)

        if len(dataset_for_train) != self.fix_length:
            raise RuntimeError(f"dataset_for_train length is not {self.fix_length}")

        dataset_for_train_total = {
            part_attribute_name: torch.stack([ identity_or_create_tensor(part[part_attribute_name]) for part in dataset_for_train ], dim=0)
            for part_attribute_name, part_atttibute_dim in self.basic_shape_structure
        }

        prediction_part = {
            key : identity_or_create_tensor(prediction_part[key])
            for key in prediction_part.keys()
        }

        return index, dataset_for_train_total, prediction_part



