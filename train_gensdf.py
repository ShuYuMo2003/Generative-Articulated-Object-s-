import yaml
import torch
import random
import trimesh
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.logging import Log, console
from utils import to_cuda

from gensdf.dataset import GenSDFDataset
from gensdf.model import SDFModulationModel
from gensdf.utils import mesh as MeshUtils
from gensdf.utils import generate_mesh_screenshot

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', dest='config',
                    help=('config file.'), required=True)

args = parser.parse_args()
config = yaml.safe_load(open(parser.parse_args().config).read())
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['usewandb']:
    import wandb
    run = wandb.init(
        project="Pointnet encoder N ONet decoder",
        config=config,
        entity="shuyumo1"
    )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(config['seed'])

# Define and Create Directories
strtime = datetime.now().strftime(r'%m-%d-%H-%M-%S')
checkpoint_output = Path(config['checkpoint_output']) / strtime
checkpoint_output.mkdir(exist_ok=True, parents=True)

eval_mesh_output_path = Path(config['eval_mesh_output_path'])
eval_mesh_output_path.mkdir(exist_ok=True, parents=True)


# Set Up Data Loaders
dl_config = config['dataloader']
dataloader = [
    DataLoader(
        GenSDFDataset(
                dataset_dir=Path(dl_config['dataset_dir']), train=train,
                samples_per_mesh=dl_config['samples_per_mesh'], pc_size=dl_config['pc_size'],
                uniform_sample_ratio=dl_config['uniform_sample_ratio'],
                cache_size=dl_config['cache_size']
            ),
        batch_size=dl_config['batch_size'], num_workers=dl_config['n_workers'],
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
    )
    for train in [True, False]
]
# dataloader[0] : train
# dataloader[1] : test

Log.info('Train Dataset Total = %s.', len(dataloader[0].dataset))
Log.info('Test Dataset Total = %s.', len(dataloader[1].dataset))

# Model
model = SDFModulationModel(config).to(config['device'])

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['sdf_lr'])

# Training Loop
for epoch_idx in range(config['num_epochs']):
    Log.info("Training Epoch = %s", epoch_idx)

    model.train()
    losses = []
    for batch, batched_data in tqdm(enumerate(dataloader[0]),
                                    desc=f'Training Epoch = {epoch_idx}', total=len(dataloader[0])):
        batched_data = to_cuda(batched_data)

        output = model(batched_data, epoch_idx)
        if output is None:
            continue
        loss, batched_recon_latent = output

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Generate Mesh and its screenshot for last batch
    if epoch_idx % config['evaluation']['freq'] == 0:
        model.eval()
        screenshots = []
        evaluation_count = min(config['evaluation']['count'], batched_recon_latent.shape[0])
        if config['evaluation']['count'] > batched_recon_latent.shape[0]:
            Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')
        for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {epoch_idx}'):
            recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
            output_mesh = (eval_mesh_output_path / f'mesh_{epoch_idx}_{batch}.ply').as_posix()
            MeshUtils.create_mesh(model.sdf_model, recon_latent,
                            output_mesh, N=config['evaluation']['resolution'],
                            max_batch=config['evaluation']['max_batch'],
                            from_plane_features=True)
            mesh = trimesh.load(output_mesh)
            # Log.debug('Loaded Mesh %s', mesh)
            screenshot = generate_mesh_screenshot(mesh)
            screenshots.append(screenshot)
        image = np.concatenate(screenshots, axis=1)

    Log.info(f"Training Loss = {np.mean(losses):.5f} on Ep {epoch_idx}")

    if config['usewandb'] and epoch_idx > 0:
        logs = {
            'train_loss': np.mean(losses)
        }
        if epoch_idx % config['evaluation']['freq'] == 0:
            logs['image'] = wandb.Image(image)
        wandb.log(logs)

    if epoch_idx % config['save_checkpoint_freq'] == 0 and epoch_idx != 0:
        checkpoint_path = checkpoint_output / f'{epoch_idx}.pth'
        torch.save(model, checkpoint_path.as_posix())