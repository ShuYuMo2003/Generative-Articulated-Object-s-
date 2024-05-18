from dataset import PartnetMobilityDataset
from pathlib import Path
from torch.utils.data import DataLoader
from glob import glob
import torch
from encoder_latent import Encoder

check_point_path = '/home/shuyumo/research/GAO/point2sdf/ckpt/sgd-e-d-201-0.9589649438858032.ckpt'
dataset_root_path = '/home/shuyumo/research/GAO/point2sdf/output'
checkpoint_output = '/home/shuyumo/research/GAO/point2sdf/ckpt'
latent_code_output_path = '/home/shuyumo/research/GAO/point2sdf/output/generation_latent_code'
Path(latent_code_output_path).mkdir(exist_ok=True)

dataset_path = list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
))

dataset = PartnetMobilityDataset(dataset_path, train_ratio=1, train=True, return_path_name=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

encoder         = Encoder(z_dim=128, c_dim=0, leaky=0.02).to(device)

checkpoint_state = torch.load(check_point_path)
encoder.load_state_dict(checkpoint_state['encoder'])

with torch.no_grad():
    for idx, (cp, sp, occ, path) in enumerate(dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)
        path = path[0]

        mean_z, logstd_z = encoder(sp, occ)

        latent = mean_z[0, ...]

        print(latent.shape)

        latent.cpu().numpy().tofile(str(Path(latent_code_output_path) / f'latent_{path}.npz'))