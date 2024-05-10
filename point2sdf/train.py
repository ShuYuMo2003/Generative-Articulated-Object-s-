import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch import distributions
from torch.nn import functional as F
from glob import glob
from pathlib import Path

from decoder import Decoder
from pointnet_encoder import SimplePointnet as Condition_Encoder
from encoder_latent import Encoder
from dataset import PartnetMobilityDataset
from visualization import visualization_as_pointcloud
from generate_3d import Generator3D

import wandb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_loss_n_acc(encoder, decoder, sp, occ, visualization=False):
    mean_z, logstd_z = encoder(sp, occ)
    q_z = distributions.Normal(mean_z, torch.exp(logstd_z))
    z = q_z.rsample()
    p0_z = distributions.Normal(
        torch.zeros(_.z_dim, device=device),
        torch.ones(_.z_dim, device=device)
    )
    kl = distributions.kl_divergence(q_z, p0_z).sum(dim=-1)
    loss = kl.mean()

    if visualization:
        visualization_as_pointcloud(decoder, z, None, device)

    logits = decoder(sp, z)
    loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
    loss = loss + loss_i.sum(-1).mean()

    acc = ((logits > 0) == occ).float().mean()

    return loss, acc

def validate(encoder, decoder, val_dataloader, visualization=False):
    losses = []
    accurancys = []
    with torch.no_grad():
        for batch, (cp, sp, occ) in enumerate(val_dataloader):
            cp = cp.to(device)
            sp = sp.to(device)
            occ = occ.to(device)

            loss, accurancy = compute_loss_n_acc(encoder, decoder, sp, occ, visualization)

            losses.append(loss)
            accurancys.append(accurancy)
    return {
        'loss': torch.tensor(losses).mean(),
        'acc': torch.tensor(accurancys).mean()
    }

run = wandb.init(
    project="Pointnet encoder N ONet decoder",
    config=dict(
        batch_size = 8,
        lr_rate = 4e-3,
        total_epoch = 2001,
        train_ratio = 0.9,
        seed = hash('ytq') & ((1 << 32) - 1),
        dataset_root_path = '/home/shuyumo/research/GAO/point2sdf/output',
        checkpoint_output = '/home/shuyumo/research/GAO/point2sdf/ckpt',
        z_dim = 128,
        leaky = 0.02
    )
)
_ = run.config


dataset_path = list(zip(
    glob(_.dataset_root_path + '/pointcloud/*'),
    glob(_.dataset_root_path + '/point/*')
))


setup_seed(_.seed)

train_dataset = PartnetMobilityDataset(dataset_path,
                                       train_ratio=_.train_ratio, train=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=_.batch_size, shuffle=True)

val_dataset = PartnetMobilityDataset(dataset_path,
                                       train_ratio=_.train_ratio, train=False)
val_dataloader = DataLoader(val_dataset,
                            batch_size=_.batch_size, shuffle=True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

encoder         = Encoder(z_dim=_.z_dim, c_dim=0, leaky=_.leaky).to(device) # unconditional
decoder         = Decoder(z_dim=_.z_dim, c_dim=0, leaky=_.leaky).to(device) # unconditional
generator       = Generator3D(decoder)

optimizer       = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=_.lr_rate
)

losses = []

print('training on device = ', device)
print('train dataset length = ', len(train_dataset))
print('valda dataset length = ', len(val_dataset))

for epoch in range(_.total_epoch):
    encoder.train()
    decoder.train()

    batch_loss = []
    batch_acc = []
    for batch, (cp, sp, occ) in enumerate(train_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        loss, acc = compute_loss_n_acc(encoder, decoder, sp, occ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    validation_dict = validate(encoder, decoder, val_dataloader, visualization=False)

    wandb.log({
        'val_acc' : validation_dict['acc'],
        'val_loss' : validation_dict['loss'],
        'train_loss' : torch.tensor(batch_loss).mean(),
        'train_acc' : torch.tensor(batch_acc).mean()

    })
    print(f'epoch {epoch} loss = {torch.tensor(batch_loss).mean()}')
    if epoch % 200 == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
        }, str(Path(_.checkpoint_output) / f'e-d-{epoch}.ckpt'))
