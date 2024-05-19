import yaml
import wandb
import torch
import random
import argparse

import numpy as np
from rich import print
from torch.utils.data import DataLoader

from transformer.dataset import get_dataset
from transformer.model import get_decoder


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(hash('Love ytq forever.') & ((1 << 20) - 1))

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('yaml configuration files to use'),
                    type=argparse.FileType('r'), required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read())
config.update({
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

dataset = get_dataset(config)
dataloader = DataLoader(dataset, **config['dataloader']['args'])
decoder = get_decoder(config)

def train(config):
    from transformer.action.trainer import Trainer
    print("start to train with config: ", config)
    train_args = config['action']['args']
    train_args.update({
        'model': decoder,
        'dataloader': dataloader
    })
    wandb_instance = wandb.init(project='transformer', config=config) if config['usewandb'] else None
    trainer = Trainer(config=config, wandb_instance=wandb_instance, **train_args)
    trainer()

if __name__ == '__main__':
    if config['action']['type'] == 'train':
        train(config)







