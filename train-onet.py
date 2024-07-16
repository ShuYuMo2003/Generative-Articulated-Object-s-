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
from torch import distributions
from torch.nn import functional as F
from onet_v2.utils.generate_3d import Generator3D
from onet_v2.dataset import PartnetMobilityDataset
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
        # mesh.export('logs/temp-validate.obj')
        plotter = pv.Plotter(off_screen=True)
        try:
            # pv_mesh = pv.read('logs/temp-validate.obj')
            plotter.add_mesh(mesh)
        except ValueError:
            print('Error in plotter.add_mesh with mesh = ', mesh)
            pass
        plotter.show()
        screenshot = plotter.screenshot()
        return screenshot

def validate(onet, val_dataloader):
    latents = []
    losses = []
    accurancys = []
    with torch.no_grad():
        for batch, (enc_sp, enc_occ, dec_sp, dec_occ) in enumerate(val_dataloader):
            enc_sp = enc_sp.to(device)
            enc_occ = enc_occ.to(device)
            dec_sp = dec_sp.to(device)
            dec_occ = dec_occ.to(device)

            loss, accurancy, mean_z = compute_loss_n_acc(onet, enc_sp, enc_occ, dec_sp, dec_occ)

            latents.append(mean_z)

            losses.append(loss)
            accurancys.append(accurancy)

    img = gen_image_from_latent(onet.get_decoder(), latents[0][[0], ...])
    return {
        'loss': torch.tensor(losses).mean(),
        'acc': torch.tensor(accurancys).mean(),
        'img': img
    }

run = wandb.init(
    project="Pointnet encoder N ONet decoder",
    config=config
)
_ = run.config

Path(_.checkpoint_output).mkdir(exist_ok=True)

setup_seed(str2hash(_.seed) & ((1 << 20) - 1))

train_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio,
                                       selected_categories=_.selected_categories, train=True)
for dd in train_dataset:
    (enc_sp, enc_occ, dec_sp, dec_occ) = dd
    print(enc_sp.shape, enc_occ.shape, dec_sp.shape, dec_occ.shape)
    break
exit(0)
train_dataloader = DataLoader(train_dataset, batch_size=_.batch_size, shuffle=True)

val_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio,
                                     selected_categories=_.selected_categories, train=True) # deprecated valiate.
val_dataloader = DataLoader(val_dataset, batch_size=_.batch_size, shuffle=True)

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
print('valda dataset length = ', len(val_dataset))
best_val_acc = -1
for epoch in tqdm(range(_.total_epoch), desc="Training"):
    onet.train()

    batch_loss = []
    batch_acc = []

    for batch, (enc_sp, enc_occ, dec_sp, dec_occ) in tqdm(enumerate(train_dataloader), desc="Batch", total=len(train_dataloader)):
        # print('batch = ', batch)

        enc_sp = enc_sp.to(device)
        enc_occ = enc_occ.to(device)
        dec_sp = dec_sp.to(device)
        dec_occ = dec_occ.to(device)
        torch.save(enc_sp, 'logs/enc_sp.pt')
        torch.save(enc_occ, 'logs/enc_occ.pt')
        torch.save(dec_sp, 'logs/dec_sp.pt')
        torch.save(dec_occ, 'logs/dec_occ.pt')
        # print(enc_sp.shape, enc_occ.shape, dec_sp.shape, dec_occ.shape)

        # print('moved to gpu')

        loss, acc, mean_z = compute_loss_n_acc(onet, enc_sp, enc_occ, dec_sp, dec_occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    validation_dict = validate(onet, train_dataloader)

    wandb.log({
        'val_acc' : validation_dict['acc'],
        'val_loss' : validation_dict['loss'],
        'train_loss' : torch.tensor(batch_loss).mean(),
        'train_acc' : torch.tensor(batch_acc).mean(),
        'val_img' : wandb.Image(validation_dict['img'], caption='validation image: val[0]'),
    })
    print(f'epoch {epoch} loss = {torch.tensor(batch_loss).mean()}')
    if best_val_acc < validation_dict['acc'] or epoch % 100 == 0:
        print('save from best_val_acc =', best_val_acc, ' to ', validation_dict['acc'], 'epoch = ', epoch)
        best_val_acc = validation_dict['acc']

        savepath = str(Path(_.checkpoint_output) / f'{epoch}-{best_val_acc}.ptn')
        torch.save(onet, savepath)
