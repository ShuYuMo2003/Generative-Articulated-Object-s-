from generate_3d import Generator3D
from decoder import Decoder
import torch
from glob import glob
from dataset import PartnetMobilityDataset
from torch.utils.data import DataLoader
from encoder_latent import Encoder
from pathlib import Path

dataset_root_path = '/home/shuyumo/research/GAO/point2sdf/output'
checkpoint_output = '/home/shuyumo/research/GAO/point2sdf/ckpt'
mesh_output_path = '/home/shuyumo/research/GAO/point2sdf/output/generation_mesh'
Path(mesh_output_path).mkdir(exist_ok=True)

train_ratio = 0.9

dataset_path = list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
))

val_dataset = PartnetMobilityDataset(dataset_path, train_ratio=train_ratio, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


device = 'cpu' # ('cuda' if torch.cuda.is_available() else 'cpu')
decoder         = Decoder(z_dim=128, c_dim=0, leaky=0.02).to(device) # unconditional
encoder         = Encoder(z_dim=128, c_dim=0, leaky=0.02).to(device)
generator       = Generator3D(device=device)

checkpoint_state = torch.load(checkpoint_output + '/e-d-400.ckpt')
decoder.load_state_dict(checkpoint_state['decoder'])
encoder.load_state_dict(checkpoint_state['encoder'])

decoder.eval()
encoder.eval()


with torch.no_grad():
    for idx, (cp, sp, occ) in enumerate(val_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        mean_z, logstd_z = encoder(sp, occ)
        mesh = generator.generate_from_latent(decoder, mean_z)
        mesh.export(str(Path(mesh_output_path) / f'{idx}.obj'))
        print(type(mesh), mesh)

