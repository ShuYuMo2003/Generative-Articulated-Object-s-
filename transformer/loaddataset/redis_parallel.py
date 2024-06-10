import copy
import redis
import pickle
import torch

import numpy as np
from rich import print
from tqdm import tqdm

from torch.utils.data import Dataset

from transformer.utils import str2hash

def identity_or_create_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class FromRedisParallelDataset(Dataset):
    def __init__(self, part_structure, host, port, db, fix_length, seed=str2hash('ytq') & ((1 << 20) - 1)):
        super().__init__()
        self.fix_length = fix_length
        self._r = redis.Redis(host=host, port=port, db=db)
        self.info = tuple(filter(lambda x : x != 'latent_generator_encoder', map(lambda x : x.decode('utf-8'), self._r.keys())))

        self.basic_shape_structure = {'dfn': 1, 'dfn_fa': 1}
        self.basic_shape_structure.update(part_structure['non_latent_info'])
        self.basic_shape_structure.update(part_structure['latent_info'])
        self.basic_shape_structure.update(part_structure['other_info'])

        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.s_part = self.generate_random_part();  self.s_part['dfn'], self.s_part['dfn_fa'] = 0,          0
        self.e_part = self.generate_random_part();  self.e_part['dfn'], self.e_part['dfn_fa'] = fix_length, fix_length

        self.onet_encoder_path = self._r.get('latent_generator_encoder').decode('utf-8')
        self.onet_decoder_path = self.onet_encoder_path.replace('-encoder.ckpt', '-decoder.ckpt')

    def generate_random_part(self):
        return { key : torch.randn(dim) for key, dim in self.basic_shape_structure.items() }

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        shape_key = self.info[index]
        shape_bytedata = self._r.get(shape_key)

        if not shape_bytedata:
            raise RuntimeError(f"key {shape_bytedata} not found")

        shape_data = pickle.loads(shape_bytedata)

        assert isinstance(shape_data, dict)

        S = copy.deepcopy(self.s_part)
        E = copy.deepcopy(self.e_part)

        raw_input = [S]
        for part in shape_data['part']:
            raw_input.append(part)

        for i in range(self.fix_length - len(raw_input)):
            raw_input.append(E)

        raw_output = raw_input[1:] + [E]

        key_padding_mask = torch.tensor([1] + [1] * len(shape_data['part']) +
                                        [0] * (self.fix_length - len(shape_data['part']) - 1), dtype=torch.int16)

        # convert to: attribute_name * (part_idx==fix_length) * attribute_dim
        input = {
            part_attribute_name: torch.stack([ identity_or_create_tensor(part[part_attribute_name])
                                              for part in raw_input], dim=0)
            for part_attribute_name in self.basic_shape_structure.keys()
        }

        # print('inp shape:', input)

        output = {
            part_attribute_name: torch.stack([ identity_or_create_tensor(part[part_attribute_name])
                                              for part in raw_output], dim=0)
            for part_attribute_name in self.basic_shape_structure.keys()
        }

        # print('oup shape:', output)

        enc_data = shape_data['enc_data']
        enc_text = shape_data['enc_text']

        enc_data.requires_grad = False

        return {
            'index': index,
            'input': input,
            'output': output,
            'key_padding_mask': key_padding_mask,
            'enc_data': enc_data,
            'enc_text': enc_text
        }





