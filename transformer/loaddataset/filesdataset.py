import torch
import json
import random
import numpy as np
import copy
from rich import print
from glob import glob
from pathlib import Path
from torch.utils.data import dataset

class FileSysDataset(dataset.Dataset):
    def __init__(self, dataset_path: str, description_for_each_file: int, cut_off: int, enc_data_fieldname: str):
        self.dataset_root_path = Path(dataset_path)

        assert enc_data_fieldname in ['description', 'image']
        self.enc_data_fieldname = enc_data_fieldname

        # import meta.json
        self.meta = json.loads((self.dataset_root_path / 'meta.json').read_text())
        self.max_count_token = self.meta['max_count_token']
        self.start_token = torch.tensor(self.meta['start_token'], dtype=torch.float32)
        self.end_token = torch.tensor(self.meta['end_token'], dtype=torch.float32)
        self.pad_token = torch.tensor(self.meta['pad_token'], dtype=torch.float32)

        # get all json files
        all_json_files = glob(str(self.dataset_root_path / '*.json'))
        all_json_files = list(filter(lambda x: 'meta.json' not in x, all_json_files))
        self.files_path = [
                (desc_idx, file)
                for desc_idx in range(description_for_each_file)
                for file in all_json_files
            ]

        random.seed(0)
        random.shuffle(self.files_path)

        if cut_off > 0:
            self.files_path = self.files_path[:cut_off]

        self.cut_off = cut_off

    def get_onet_ckpt_path(self):
        return self.meta['best_ckpt_path']

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        # print('index: ', index)
        current_enc_idx, file_path = self.files_path[index]
        data = json.loads(Path(file_path).read_text())
        # print('index = ', index)
        # print('file_path = ', file_path)
        enc_path = data[self.enc_data_fieldname][current_enc_idx]
        enc = np.load(enc_path, allow_pickle=True)

        if self.enc_data_fieldname == 'description':
            enc = enc.item()

        total_token = len(data['exist_node'])
        assert len(data['exist_node']) == len(data['inferenced_token'])

        # Process Input
        input = data['exist_node']
        # with open('input.json', 'w') as f:
        #     json.dump(input, f, indent=4)

        for node_idx, node in enumerate(input):
            node['token'] = torch.tensor(node['token'], dtype=torch.float32)
            dfn_fa = node['dfn_fa']
            for idx in range(len(input)):
                if input[idx]['dfn'] == dfn_fa:
                    node['fa'] = idx
                    break
            assert 'fa' in node, f"Can't find father node for {node['dfn']}"
            assert node['fa'] <= node_idx

        for node in input:
            if node.get('dfn') is not None: del node['dfn']
            if node.get('dfn_fa') is not None: del node['dfn_fa']

        for _ in range(self.max_count_token - len(input)):
            input.append({'token': copy.deepcopy(self.pad_token), 'fa': 0})

        transformed_input = {
                'token': torch.stack([node['token'] for node in input]),
                'fa': torch.tensor([node['fa'] for node in input], dtype=torch.int)
            }

        # Process Output
        infer_nodes = data['inferenced_token']
        output = []
        # 1:   not end token,    0: end token
        output_skip_end_token_mask = []
        for node in infer_nodes:
            output.append(torch.tensor(node['token'], dtype=torch.float32))
            output_skip_end_token_mask.append(0 if node['dfn'] == -1 else 1)

        for _ in range(self.max_count_token - len(output)):
            output.append(copy.deepcopy(self.pad_token))
            output_skip_end_token_mask.append(1)

        output = torch.stack(output)

        output_skip_end_token_mask = torch.tensor(output_skip_end_token_mask, dtype=torch.int)

        # Process Padding Mask
        padding_mask = torch.ones(self.max_count_token, dtype=torch.int16)
        padding_mask[total_token:] = 0


        # with open(f'logs/debug/output{index}.json', 'w') as f:
        #     f.write(d)

        # print('index = ', index)
        # print('file_path = ', file_path)
        # print('transformed_input[token] = ', transformed_input['token'].shape)
        # print('transformed_input[fa] = ', transformed_input['fa'].shape)
        # print('output = ', output.shape)
        # print('mask = ', mask.shape)
        # print('desc[encoded_text] = ', desc['encoded_text'].shape)
        # print('desc[text] = ', type(desc['text']), desc['text'])

        # if self.cut_off > 0:
        #     print('index = ', index)
        #     print('padding_mask = ', padding_mask)
        #     print('output_skip_end_token_mask = ', output_skip_end_token_mask)

        # print('enc', enc)
        # print(self.enc_data_fieldname, self.enc_data_fieldname == 'description')

        return [transformed_input, output, padding_mask, output_skip_end_token_mask] +   \
                    ([enc['encoded_text'], enc['text']] if self.enc_data_fieldname == 'description'
                else [enc.astype(np.float32), str(enc_path)])