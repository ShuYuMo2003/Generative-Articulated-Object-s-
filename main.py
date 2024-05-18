import torch
import argparse
import yaml

from torch.utils.data import DataLoader

from transformer.dataset import get_dataset
from transformer.model import get_decoder


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('yaml configuration files to use'),
                    type=argparse.FileType('r'), required=True)
args = parser.parse_args()
config = yaml.safe_load(args.config.read())
config.update({
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

dataloader = DataLoader(dataset=get_dataset(config), **config['DataLoader'])
decoder = get_decoder(config)

def train(config):
    from transformer.action.trainer import Trainer
    train_args = config['action']['args']
    train_args.update({
        'model': decoder,
        'dataloader': dataloader,
        'device': config['device'],
    })
    trainer = Trainer(**train_args)
    trainer()









