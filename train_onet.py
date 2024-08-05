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
from onet.utils.generator_sim import Generator3DSimple
from onet.onet import ONet

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

def compute_loss_n_acc(onet, enc_sp, enc_occ_or_sdf, dec_sp, dec_occ_or_sdf):
    if not _.sdf_dataset:
        logits, kl_loss, mean_z = onet(enc_sp, enc_occ_or_sdf, dec_sp)
        i_loss = F.binary_cross_entropy_with_logits(
                logits, dec_occ_or_sdf, reduction='none')
        loss = (_.sdf_ratio) * i_loss.sum(-1).mean() + (1 - _.sdf_ratio) * kl_loss
        acc = ((logits > 0) == dec_occ_or_sdf).float().mean()
    else:
        sdf, kl_loss, mean_z = onet(enc_sp, enc_occ_or_sdf, dec_sp)
        i_loss = F.mse_loss(sdf, dec_occ_or_sdf, reduction='none')
        loss = (_.sdf_ratio) * i_loss.sum(-1).mean() + (1 - _.sdf_ratio) * kl_loss
        # print('sdf_loss =', (_.sdf_ratio) * i_loss.sum(-1).mean(), 'kl_loss =', (1 - _.sdf_ratio) * kl_loss)
        acc = ((sdf.abs() < 0.01) == (dec_occ_or_sdf.abs() < 0.01)).float().mean()
    return loss, acc, mean_z

def gen_image_from_latent(decoder, mean_z):
    screenshots = []
    for generator in tqdm(generators, 'evaluating mesh'):
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
        screenshots.append(screenshot)
    screenshot = np.concatenate(screenshots, axis=1)
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
    from onet.dataset import PartnetMobilityDataset
    train_dataset = PartnetMobilityDataset(_.dataset_root_path, train_ratio=_.train_ratio,
                                        selected_categories=_.selected_categories, train=True, sdf_dataset=_.sdf_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=_.batch_size, shuffle=True, num_workers=12)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

print('running on device = ', device)

onet        = ONet(dim_z=_.z_dim, emb_sigma=_.emb_sigma).to(device)
generators = [
    Generator3DSimple(device=device, threshold=threshold)
    for threshold in ([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if not _.sdf_dataset
                 else [-0.02, -0.01, 0, 0.01, 0.02])
]

# @from: https://nlp.seas.harvard.edu/annotated-transformer/#batches-and-masking
def lr_rate_func(step, factor, warmup, model_size=512):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

# print(config['optimizer'])
# print(config['optimizer']['betas'])
# print(config['optimizer']['eps'])
optimizer = torch.optim.Adam(onet.parameters(), lr=1, betas=config['optimizer']['betas'], eps=config['optimizer']['eps'])
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step:
                                        lr_rate_func(step, config['optimizer']['scheduler_factor'],
                                                        config['optimizer']['scheduler_warmup']))

losses = []

print('training on device = ', device)
print('train dataset length = ', len(train_dataset))
best_val_acc = -1
for epoch in tqdm(range(_.total_epoch), desc="Training"):
    onet.train()

    batch_loss = []
    batch_acc = []

    for batch, (enc_sp, enc_occ_or_sdf, dec_sp, dec_occ_or_sdf) in tqdm(enumerate(train_dataloader), desc="Batch", total=len(train_dataloader)):
        enc_sp = enc_sp.to(device)
        enc_occ_or_sdf = enc_occ_or_sdf.to(device)
        dec_sp = dec_sp.to(device)
        dec_occ_or_sdf = dec_occ_or_sdf.to(device)

        loss, acc, mean_z = compute_loss_n_acc(onet, enc_sp, enc_occ_or_sdf, dec_sp, dec_occ_or_sdf)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    lr_scheduler.step()

    img = gen_image_from_latent(onet.get_decoder(), mean_z[[0], ...])

    info = {
        'train_loss' : torch.tensor(batch_loss).mean(),
        'train_acc' : torch.tensor(batch_acc).mean(),
        'val_img' : wandb.Image(img, caption='validation image: val[0]'),
        'lr': optimizer.param_groups[0]['lr']
    }

    wandb.log(info)
    print(f'epoch {epoch} ', info)
    if epoch % 50 == 0:
        acc = torch.tensor(batch_acc).mean().item()
        savepath = str(Path(_.checkpoint_output) / f'{acc}-{epoch}.ptn')
        torch.save(onet, savepath)
