import yaml
import torch
import wandb
import trimesh
import random
from datetime import datetime
import argparse

import numpy as np
import pyvista as pv
from tqdm import tqdm, trange
from pathlib import Path
from rich import print

from torch.utils.data import DataLoader
from torch.nn import functional as F
from onet.utils.generator_sim import Generator3DSimple
from onet.utils.generator import Generator3DSDF
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
    if not _.train_by_sdf:
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
        acc = ((sdf < 0) == (dec_occ_or_sdf < 0)).float().mean()
    return loss, acc, mean_z

def gen_image_from_latent(sdf_generator, mean_z, epoch):
    mean_z = mean_z.detach()
    screenshots = []
    for batch_idx in trange(mean_z.size(0), desc='evaluating mesh'):
        stats_dict = dict()
        mesh = sdf_generator.generate_from_latent(mean_z[[batch_idx], ...], stats_dict=stats_dict)
        print(stats_dict)
        if mesh.vertices.shape[0] == 0:
            continue
        mesh.export((eval_mesh_output_path / f'{epoch}_{batch_idx}.obj').as_posix(), file_type='obj')
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh)
        plotter.show()
        screenshot = plotter.screenshot()
        screenshots.append(screenshot)
    if len(screenshots) == 0:
        return np.zeros((10, 10, 3))
    screenshot = np.concatenate(screenshots, axis=1)
    return screenshot

run = wandb.init(
    project="Pointnet encoder N ONet decoder",
    config=config,
    entity="shuyumo1"
)
_ = run.config

# create checkpoint output directory
strtime = datetime.now().strftime(r'%m-%d-%H-%M-%S')
checkpoint_output = Path(_.checkpoint_output) / strtime
checkpoint_output.mkdir(exist_ok=True, parents=True)

eval_mesh_output_path = Path(_.eval_mesh_output_path)
eval_mesh_output_path.mkdir(exist_ok=True, parents=True)

# set seed
setup_seed(str2hash(_.seed) & ((1 << 20) - 1))

# load dataset
from onet.dataset import PartnetMobilityDataset
train_dataset = PartnetMobilityDataset(_.dataset_root_path, train=True, sdf_dataset=_.train_by_sdf)
train_dataloader = DataLoader(train_dataset, batch_size=_.batch_size, shuffle=True, num_workers=18)

val_dataset = PartnetMobilityDataset(_.dataset_root_path, train=False, sdf_dataset=_.train_by_sdf)
val_dataloader = DataLoader(val_dataset, batch_size=_.batch_size, shuffle=False, num_workers=18)

# set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('running on device = ', device)

# set up model
onet = ONet(dim_z=_.z_dim, emb_sigma=_.emb_sigma).to(device)

# set up generator for visualization
sdf_generator = Generator3DSDF(model=onet.get_decoder(), device=device, threshold=0, positive_inside=True,
                        refinement_step=40, simplify_nfaces=6000, upsampling_steps=5)

# set up optimizer.
# @from: https://nlp.seas.harvard.edu/annotated-transformer/#batches-and-masking
def lr_rate_func(step, factor, warmup, model_size=512):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

if config['optimizer']['use_warmup']:
    optimizer = torch.optim.Adam(onet.parameters(), lr=1,
                                betas=config['optimizer']['betas'],
                                eps=config['optimizer']['eps'])

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lambda step:lr_rate_func(
                    step,
                    config['optimizer']['scheduler_factor'],
                    config['optimizer']['scheduler_warmup'])
        )
else:
    optimizer = torch.optim.Adam(onet.parameters(), lr=config['optimizer']['lr'])
    lr_scheduler = None

print('training on device = ', device)
print('train dataset length = ', len(train_dataset))
print('val dataset length = ', len(val_dataset))
img = np.zeros((10, 10, 3))
for epoch in tqdm(range(_.total_epoch), desc="Training"):
    # train
    train_batch_loss = []
    train_batch_acc = []
    onet.train()
    for batch, batched_data in tqdm(enumerate(train_dataloader),
                                    desc="Batch", total=len(train_dataloader)):
        batched_data = list(map(lambda x: x.to(device), batched_data))

        loss, acc, mean_z_train = compute_loss_n_acc(onet, *batched_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_batch_loss.append(loss.item())
        train_batch_acc.append(acc.item())

    # validation
    validate_batch_loss = []
    validate_batch_acc = []
    onet.eval()
    for batch, batched_data in tqdm(enumerate(val_dataloader),
                                    desc="Validation", total=len(val_dataloader)):
        batched_data = list(map(lambda x: x.to(device), batched_data))

        with torch.no_grad():
            loss, acc, mean_z = compute_loss_n_acc(onet, *batched_data)

        validate_batch_loss.append(loss.item())
        validate_batch_acc.append(acc.item())

    if lr_scheduler is not None:
        lr_scheduler.step()

    if epoch % 10 == 0:
        img = gen_image_from_latent(sdf_generator, mean_z_train[:5], epoch)

    info = {
        'train_loss' : torch.tensor(train_batch_loss).mean(),
        'train_acc' : torch.tensor(train_batch_acc).mean(),
        'val_loss' : torch.tensor(validate_batch_loss).mean(),
        'val_acc' : torch.tensor(validate_batch_acc).mean(),
        'img' : wandb.Image(img, caption='validation image: val[0]'),
        'lr': optimizer.param_groups[0]['lr']
    }

    wandb.log(info)
    print(f'epoch {epoch} ', info)
    if epoch % 20 == 0:
        acc = torch.tensor(train_batch_acc).mean().item()
        savepath = (checkpoint_output / f'{epoch}-{acc}.ptn').as_posix()
        torch.save(onet, savepath)