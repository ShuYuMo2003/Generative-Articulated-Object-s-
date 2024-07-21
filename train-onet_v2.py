import yaml
import torch
import wandb
import random
import argparse

import numpy as np
import pyvista as pv
from tqdm import tqdm
from pathlib import Path
from rich import print

from torch.utils.data import DataLoader
from torch.nn import functional as F
from onet_v2.utils.generate_3d import Generator3D
from onet_v2.onet import ONet

from transformer.utils import str2hash

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('config file.'), required=True)
config = yaml.safe_load(open(parser.parse_args().config).read())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_loss_n_acc(onet, enc_sp, enc_occ, dec_sp, dec_occ):
    logits, kl_loss, mean_z = onet(enc_sp, enc_occ, dec_sp)
    i_loss = F.binary_cross_entropy_with_logits(
            logits, dec_occ, reduction='none')
    loss = i_loss.sum(-1).mean() + kl_loss
    acc = ((logits > 0) == dec_occ).float().mean()
    return loss, acc, mean_z

def gen_image_from_latent(decoder, mean_z):
    with torch.no_grad():
        mesh = generator.generate_from_latent(decoder, mean_z)
        plotter = pv.Plotter(off_screen=True)
        try:
            plotter.add_mesh(mesh)
        except ValueError:
            print('Error in plotter.add_mesh with mesh = ', mesh)
            pass
        plotter.show()
        screenshot = plotter.screenshot()
        return screenshot

run = wandb.init(
    project="Pointnet encoder N ONet decoder",
    config=config,
    entity="shuyumo1"
)
_ = run.config

Path(_.checkpoint_output).mkdir(exist_ok=True)

setup_seed(str2hash(_.seed) & ((1 << 20) - 1))

if _.use_v1_dataset:
    from onet.dataset import PartnetMobilityDataset
    train_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio, train=True)
    print('load v1 dataset')
else:
    from onet_v2.dataset import PartnetMobilityDataset
    train_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio,
                                        selected_categories=_.selected_categories, train=True)

train_dataloader = DataLoader(train_dataset, batch_size=_.batch_size, shuffle=True, num_workers=2)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

print('running on device = ', device)

onet        = ONet(dim_z=_.z_dim, emb_sigma=_.emb_sigma).to(device)
generator   = Generator3D(device=device)

if _.optimizer == 'sgd':
    optimizer = torch.optim.SGD(onet.parameters(), lr=_.lr_rate)
elif _.optimizer == 'adam':
    optimizer = torch.optim.Adam(onet.parameters(), lr=_.lr_rate)
else:
    raise ValueError(f'optimizer {_.optimizer} not supported')

losses = []

print('training on device = ', device)
print('train dataset length = ', len(train_dataset))
best_val_acc = -1
for epoch in tqdm(range(_.total_epoch), desc="Training"):
    onet.train()

    batch_loss = []
    batch_acc = []

    for batch, (enc_sp, enc_occ, dec_sp, dec_occ) in tqdm(enumerate(train_dataloader), desc="Batch", total=len(train_dataloader)):
        enc_sp = enc_sp.to(device)
        enc_occ = enc_occ.to(device)
        dec_sp = dec_sp.to(device)
        dec_occ = dec_occ.to(device)

        loss, acc, mean_z = compute_loss_n_acc(onet, enc_sp, enc_occ, dec_sp, dec_occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    img = gen_image_from_latent(onet.get_decoder(), mean_z[[0], ...])

    info = {
        'train_loss' : torch.tensor(batch_loss).mean(),
        'train_acc' : torch.tensor(batch_acc).mean(),
        'val_img' : wandb.Image(img, caption='validation image: val[0]'),
    }

    wandb.log(info)
    print(f'epoch {epoch} ', info)
    if epoch % 50 == 0:
        acc = torch.tensor(batch_acc).mean().item()
        savepath = str(Path(_.checkpoint_output) / f'{acc}-{epoch}.ptn')
        torch.save(onet, savepath)
