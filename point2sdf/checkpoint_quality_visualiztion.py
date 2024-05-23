from generate_3d import Generator3D
from decoder import Decoder
import torch
from glob import glob
from dataset import PartnetMobilityDataset
from torch.utils.data import DataLoader
from encoder_latent import Encoder
from pathlib import Path
import shutil
import numpy as np
import pyvista as pv


dataset_root_path = 'output/2_dataset'
checkpoint_output = 'ckpt'
mesh_output_path = 'output/5_visulization/'
import shutil
shutil.rmtree(mesh_output_path, ignore_errors=True)
Path(mesh_output_path).mkdir(exist_ok=True)

train_ratio = 0.9

dataset_path = list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
))

val_dataset = PartnetMobilityDataset(dataset_path, train_ratio=train_ratio, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


device = ('cuda' if torch.cuda.is_available() else 'cpu')
decoder         = Decoder(z_dim=128, c_dim=0, emb_sigma=0.1, leaky=0.02).to(device) # unconditional
encoder         = Encoder(z_dim=128, c_dim=0, emb_sigma=0.1, leaky=0.02).to(device)
generator       = Generator3D(device=device)

checkpoint_state = torch.load(checkpoint_output + '/sgd-e-d-0.1-455-0.9807546138763428.ckpt')
decoder.load_state_dict(checkpoint_state['decoder'])
encoder.load_state_dict(checkpoint_state['encoder'])

decoder.eval()
encoder.eval()

total_image = []

latentcode = []

with torch.no_grad():
    for idx, (cp, sp, occ) in enumerate(val_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        mean_z, logstd_z = encoder(sp, occ)
        print(idx, mean_z.shape, logstd_z.shape)
        mesh = generator.generate_from_latent(decoder, mean_z)
        stem_name = str(Path(mesh_output_path) / f'{idx}')
        mesh.export(stem_name + '.obj')
        shutil.copy(stem_name + '.obj', f'/mnt/d/Research/transfer/sdfs/{idx}.obj')
        latentcode.append(mean_z)
        # if idx == 6: break

torch.manual_seed(20031011)
noise = torch.randn_like(mean_z)
# start, end = latentcode[3], latentcode[3] + noise
start, end = latentcode[0], latentcode[3]

for noise_sigma in np.linspace(0, 1, 40):
    # zero_noise = torch.randn_like(mean_z)
    print('calc sigma:', noise_sigma)

    noisy_mean_z = start * (1-noise_sigma) + end * noise_sigma

    mesh = generator.generate_from_latent(decoder, noisy_mean_z)
    print(type(mesh), mesh)

    stem_name = str(Path(mesh_output_path) / f'{idx}-{noise_sigma}')

    mesh.export(stem_name + '.obj')
    shutil.copy(stem_name + '.obj', f'/mnt/d/Research/transfer/sdfs/{idx}-{noise_sigma}.obj')


    plotter = pv.Plotter()
    try:
        pv_mesh = pv.read(stem_name + '.obj')
        plotter.add_mesh(pv_mesh)
    except ValueError:
        print('Error in rendering', stem_name + '.obj')
        continue
    plotter.show()
    plotter.screenshot(stem_name + '.png')
    total_image.append((stem_name + '.png', noise_sigma))
    shutil.copy(stem_name + '.png', f'/mnt/d/Research/transfer/sdfs/{idx}-{noise_sigma}.png')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for png, sigma in total_image:
    fig, ax = plt.subplots()
    img = mpimg.imread(png)
    # 不显示坐标轴
    ax.axis('off')
    ax.imshow(img)
    ax.text(0, 0, f'sigma={sigma}', color='red', fontsize=15)
    fig.savefig(png)

import imageio
with imageio.get_writer(uri=str(Path(mesh_output_path) / 'combine.gif'), mode='I', fps=6) as writer:
    for png, sigma in total_image:
        writer.append_data(imageio.imread(png))

shutil.copy(str(Path(mesh_output_path) / 'combine.gif'), f'/mnt/d/Research/transfer/combine.gif')