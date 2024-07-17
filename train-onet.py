import yaml
import torch
import random
import argparse

import numpy as np
import pyvista as pv
from tqdm import tqdm
from glob import glob
from pathlib import Path
from rich import print

from torch.utils.data import DataLoader
from torch import distributions
from torch.nn import functional as F

from onet.decoder import Decoder
from onet.pointnet_encoder import SimplePointnet as Condition_Encoder
from onet.encoder_latent import Encoder
from onet.dataset import PartnetMobilityDataset
from onet.generate_3d import Generator3D

from transformer.utils import str2hash

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('config file.'), required=True)
config = yaml.safe_load(open(parser.parse_args().config).read())


import wandb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_loss_n_acc(encoder, decoder, enc_sp, enc_occ, dec_sp, dec_occ):
    mean_z, logstd_z = encoder(enc_sp, enc_occ)
    q_z = distributions.Normal(mean_z, torch.exp(logstd_z))
    z = q_z.rsample()
    p0_z = distributions.Normal(
        torch.zeros(_.z_dim, device=device),
        torch.ones(_.z_dim, device=device)
    )
    kl = distributions.kl_divergence(q_z, p0_z).sum(dim=-1)
    loss = kl.mean()

    logits = decoder(dec_sp, z)
    loss_i = F.binary_cross_entropy_with_logits(
            logits, dec_occ, reduction='none')
    loss = loss + loss_i.sum(-1).mean()

    acc = ((logits > 0) == dec_occ).float().mean()

    return loss, acc, mean_z

def gen_image_from_latent(decoder, mean_z):
    with torch.no_grad():
        mesh = generator.generate_from_latent(decoder, mean_z)
        mesh.export('logs/temp-validate.obj')
        plotter = pv.Plotter()
        try:
            pv_mesh = pv.read('logs/temp-validate.obj')
            plotter.add_mesh(pv_mesh)
        except ValueError:
            print('error')
            pass
        plotter.show()
        screenshot = plotter.screenshot()
        return screenshot

def validate(encoder, decoder, val_dataloader):
    latents = []
    losses = []
    accurancys = []
    with torch.no_grad():
        for batch, (enc_sp, enc_occ, dec_sp, dec_occ) in enumerate(train_dataloader):
            enc_sp = enc_sp.to(device)
            enc_occ = enc_occ.to(device)
            dec_sp = dec_sp.to(device)
            dec_occ = dec_occ.to(device)

            loss, accurancy, mean_z = compute_loss_n_acc(encoder, decoder, enc_sp, enc_occ, dec_sp, dec_occ)

            latents.append(mean_z)

            losses.append(loss)
            accurancys.append(accurancy)

    img = gen_image_from_latent(decoder, latents[0][[0], ...])
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

train_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=_.batch_size, shuffle=True)

val_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=_.batch_size, shuffle=True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

print('running on device = ', device)

encoder         = Encoder(z_dim=_.z_dim, c_dim=0, emb_sigma=_.emb_sigma).to(device) # unconditional
decoder         = Decoder(z_dim=_.z_dim, c_dim=0, emb_sigma=_.emb_sigma).to(device) # unconditional
generator       = Generator3D(device=device)

if _.optimizer == 'sgd':
    optimizer       = torch.optim.SGD(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=_.lr_rate
    )
elif _.optimizer == 'adam':
    optimizer       = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=_.lr_rate
    )
else:
    raise ValueError(f'optimizer {_.optimizer} not supported')

losses = []

print('training on device = ', device)
print('train dataset length = ', len(train_dataset))
print('valda dataset length = ', len(val_dataset))
best_val_acc = -1
for epoch in tqdm(range(_.total_epoch), desc="Training"):
    encoder.train()
    decoder.train()

    batch_loss = []
    batch_acc = []

    for batch, (enc_sp, enc_occ, dec_sp, dec_occ) in enumerate(train_dataloader):
        enc_sp = enc_sp.to(device)
        enc_occ = enc_occ.to(device)
        dec_sp = dec_sp.to(device)
        dec_occ = dec_occ.to(device)

        loss, acc, mean_z = compute_loss_n_acc(encoder, decoder, enc_sp, enc_occ, dec_sp, dec_occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    validation_dict = validate(encoder, decoder, val_dataloader)

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

        savepath = str(Path(_.checkpoint_output) / f'{epoch}-{best_val_acc}-TYPE.ckpt')
        torch.save(encoder, savepath.replace('TYPE', 'encoder'))
        torch.save(decoder, savepath.replace('TYPE', 'decoder'))