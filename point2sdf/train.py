import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torch import distributions
from torch.nn import functional as F
from glob import glob
from pathlib import Path

from tqdm import tqdm
import pyvista as pv

from decoder import Decoder
from pointnet_encoder import SimplePointnet as Condition_Encoder
from encoder_latent import Encoder
from dataset import PartnetMobilityDataset
from visualization import visualization_as_pointcloud
from generate_3d import Generator3D

parser = argparse.ArgumentParser()
parser.add_argument('--embsigma', '-e', dest='emb_sigma',
                    help=('emb_sigma'),
                    type=float, default=0.2)
args = parser.parse_args()
emb_sigma = args.emb_sigma


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

    # if visualization:
    #     visualization_as_pointcloud(decoder, z, None, device)

    logits = decoder(sp, z)
    loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
    loss = loss + loss_i.sum(-1).mean()

    acc = ((logits > 0) == occ).float().mean()

    return loss, acc, mean_z

def gen_image_from_latent(decoder, mean_z):
    # mean_z should be with batch size 1.
    with torch.no_grad():
        mesh = generator.generate_from_latent(decoder, mean_z)
        mesh.export('temp-validate.obj')
        plotter = pv.Plotter()
        try:
            pv_mesh = pv.read('temp-validate.obj')
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
        for batch, (cp, sp, occ) in enumerate(val_dataloader):
            cp = cp.to(device)
            sp = sp.to(device)
            occ = occ.to(device)

            loss, accurancy, mean_z = compute_loss_n_acc(encoder, decoder, sp, occ)

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
    name=f"with-foutier-embedding-emb_sigma={emb_sigma}",
    config=dict(
        batch_size = 8,
        lr_rate = 1.5e-3,
        total_epoch = 1000,
        train_ratio = 0.9,
        seed = hash('ytq') & ((1 << 30) - 1),
        dataset_root_path = 'output/2_dataset',
        checkpoint_output = 'ckpt',
        z_dim = 128,
        leaky = 0.05,
        emb_sigma=emb_sigma,
        optimizer = 'adam'
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

encoder         = Encoder(z_dim=_.z_dim, c_dim=0, emb_sigma=_.emb_sigma, leaky=_.leaky).to(device) # unconditional
decoder         = Decoder(z_dim=_.z_dim, c_dim=0, emb_sigma=_.emb_sigma, leaky=_.leaky).to(device) # unconditional
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

    for batch, (cp, sp, occ) in enumerate(train_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        loss, acc, mean_z = compute_loss_n_acc(encoder, decoder, sp, occ)

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
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
        }, str(Path(_.checkpoint_output) / f'sgd-e-d-{emb_sigma}-{epoch}-{best_val_acc}.ckpt'))
