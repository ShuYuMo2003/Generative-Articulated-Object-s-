from pathlib import Path
import json
from transformer.action.evaluater import Evaluater
import argparse
import yaml
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('yaml configuration files to use'),
                    type=argparse.FileType('r'), required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read())
config.update({
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

train_keys_path = Path('./dataset/train_keys.json')
train_keys = json.load(train_keys_path.open('r'))['train_keys']

prompt_ds_path = Path('./dataset/4_screenshot_description')

eval_args = config['action']['args']
evaler = Evaluater(config=config, **eval_args)

output_info_path = Path(eval_args['eval_output_path']) / '1_info'
output_mesh_path = Path(eval_args['eval_output_path']) / '2_mesh'
output_info_path.mkdir(parents=True, exist_ok=True)
output_mesh_path.mkdir(parents=True, exist_ok=True)
# clear the output folder
for file in output_info_path.iterdir():
    file.unlink()
for file in output_mesh_path.iterdir():
    file.unlink()
print('output folder cleared')

for folder in prompt_ds_path.iterdir():
    if not folder.is_dir():
        continue
    folder_name = folder.name
    if folder_name in train_keys:
        continue
    with open(folder / '0.txt', 'r') as f:
        print(f'generating with test data: {folder_name}/0.txt')
        data = f.read()
        evaler.reconstruct(data, folder_name)