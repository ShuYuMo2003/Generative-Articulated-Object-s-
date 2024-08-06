import sys
import json
import copy
import shutil
import numpy as np
import torch
from rich import print
from glob import glob
from pathlib import Path
from tqdm import tqdm

sys.path.append('..')
from onet.dataset import PartnetMobilityDataset
from utils.utils import (tokenize_part_info, generate_special_tokens,
                            HighPrecisionJsonEncoder)
from transformer.utils import str2hash

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_token = None
end_token = None
pad_token = None
max_count_token = 0
best_ckpt_path = None

def determine_latentcode_encoder():
    global best_ckpt_path

    # onets_ckpt_paths = glob('../checkpoints/onet/*.ptn')
    # onets_ckpt_paths.sort(key=lambda x: -float(x.split('/')[-1].split('-')[0]))

    # best_ckpt_path = onets_ckpt_paths[0]
    best_ckpt_path = '../checkpoints/onet_sdf/0.7569100260734558-700.ptn'
    print('Using best ckpt:', best_ckpt_path)

    onet = torch.load(best_ckpt_path)
    return onet

def evaluate_latent_codes(onet):
    dataset = PartnetMobilityDataset(
            path='../dataset/2_onet_sdf_dataset',
            train_ratio=0,
            train=False,
            selected_categories=['*'],
            return_path_name=True,
            sdf_dataset=True,
        )
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

    onet.eval()
    onet = onet.to(device)

    path_to_latent = {}

    for i, (enc_samplepoints, enc_occ, dec_samplepoints, dec_occ, path_name) in               \
                tqdm(enumerate(dataloader), desc="Evaluating latent codes", total=len(dataloader)):
        enc_samplepoints = enc_samplepoints.to(device)
        enc_occ = enc_occ.to(device)
        dec_samplepoints = dec_samplepoints.to(device)
        dec_occ = dec_occ.to(device)

        mean_z, logstd_z = onet.encoder(enc_samplepoints, enc_occ)
        # print("path_name = ", path_name)

        for batch in range(mean_z.shape[0]):
            latent = mean_z[batch, ...]
            latent_numpy = latent.detach().cpu().numpy()
            path = path_name[batch]
            path = path.split('/')[-1]
            path = Path(path).stem + ".ply"
            path_to_latent[path] = latent_numpy

    # print('path_to_latent', path_to_latent)

    return path_to_latent

def process(shape_info_path:Path, transformer_dataset_path:Path, encoded_text_paths:list[Path], path_to_latent:dict):
    global start_token, end_token, pad_token, max_count_token

    shape_info = json.loads(shape_info_path.read_text())
    meta_data = shape_info['meta']

    new_parts_info = []

    # Tokenize
    for part_info in shape_info['part']:
        # Add the latent code
        mesh_file_name = part_info['mesh']
        latent = path_to_latent.get(mesh_file_name)
        if latent is None:
            return f"[Error] Latent code not found for {mesh_file_name}"
        part_info['latent_code'] = latent.tolist()

        token = tokenize_part_info(part_info)

        new_parts_info.append({
                'token': token,
                'dfn_fa': part_info['dfn_fa'],
                'dfn': part_info['dfn'],
            })


    start_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                          str2hash('This is start token') & ((1 << 10) - 1))

    end_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                        str2hash('This is end token') & ((1 << 10) - 1))

    pad_token = generate_special_tokens(len(new_parts_info[-1]['token']),
                                        str2hash('This is pad token') & ((1 << 10) - 1))

    root = None
    for part_info in new_parts_info:
        if part_info['dfn_fa'] == 0:
            root = part_info

        part_info['child'] = list(filter(lambda x: x['dfn_fa'] == part_info['dfn'], new_parts_info))
        part_info['child'].sort(key=lambda x: x['dfn'])

    assert root is not None

    exist_node = [{'token': start_token, 'dfn': 0, 'dfn_fa' : 0, 'child': [root]}]

    datasets = []

    while True: # end and start token
        inferenced_token = []
        for node in exist_node:
            if len(node['child']) > 0:
                inferenced_token.append(node['child'][0])
                node['child'].pop(0)
            else:
                inferenced_token.append({'token': end_token, 'dfn': -1, 'dfn_fa' : -1})

        datasets.append((copy.deepcopy(exist_node), copy.deepcopy(inferenced_token)))

        all_end = True
        for node in inferenced_token:
            if node['dfn'] != -1:
                exist_node.append(node)
                all_end = False

        if all_end: break

    # TODO: save dataset.
    prefix_name = meta_data['catecory'] + '_' + meta_data['shape_id']

    # Fetch the encoded text path.
    encoded_text_paths = list(filter(lambda x: x.stem.startswith(prefix_name), encoded_text_paths))
    encoded_text_paths = list(map(lambda x : x.as_posix().replace('../', ''), encoded_text_paths))
    encoded_text_paths.sort()

    if len(encoded_text_paths) < 2:
        return f"[Error] Encoded text path not found for {prefix_name}"



    for idx, dataset in enumerate(datasets):
        dataset_name = prefix_name + '_' + str(idx) + '.json'

        for node in dataset[0]:
            if node.get('child') is not None: del node['child']
        for node in dataset[1]:
            if node.get('child') is not None: del node['child']

        max_count_token = max(max_count_token, len(dataset[0]))

        with open(transformer_dataset_path / dataset_name, 'w') as f:
            text = json.dumps({
                    'meta': meta_data,
                    'shape_info': shape_info['part'],
                    'exist_node': dataset[0],
                    'inferenced_token': dataset[1],
                    'description': encoded_text_paths,
                }, cls=HighPrecisionJsonEncoder, indent=2)
            f.write(text)

    return f"[Success] Processed {shape_info_path} part count = {len(datasets)}"

if __name__ == '__main__':
    transformer_dataset_path = Path('../dataset/4_transformer_dataset')
    shutil.rmtree(transformer_dataset_path, ignore_errors=True)
    transformer_dataset_path.mkdir(exist_ok=True)

    shape_info_paths = list(map(Path, glob('../dataset/1_preprocessed_info/*')))
    onet = determine_latentcode_encoder()
    path_to_latent = evaluate_latent_codes(onet)

    encoded_text_path = Path('../dataset/4_screenshot_description_encoded')
    encoded_text_paths = list(map(Path, glob((encoded_text_path / '*').as_posix())))

    failed = []

    for shape_info_path in tqdm(shape_info_paths, desc="Processing shape info"):
        status = process(shape_info_path, transformer_dataset_path, encoded_text_paths, path_to_latent)
        if "Success" not in status:
            failed.append((shape_info_path.stem, status))
        print(shape_info_path.as_posix(), status)

    with open(transformer_dataset_path / 'meta.json', 'w') as f:
        json.dump({
            'start_token': start_token,
            'end_token': end_token,
            'pad_token': pad_token,
            'max_count_token': max_count_token,
            'best_ckpt_path': best_ckpt_path.replace('../', ''),
        }, f, cls=HighPrecisionJsonEncoder, indent=2)

    print('Failed:', failed)
    print('Failed count:', len(failed))

