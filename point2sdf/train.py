import torch
from torch.utils.data import DataLoader
from torch import distributions
from torch.nn import functional as F
from glob import glob
from pathlib import Path

from decoder import Decoder
from pointnet_encoder import SimplePointnet as Encoder
from dataset import PartnetMobilityDataset

dataset_root_path = '/home/shuyumo/research/GAO/point2sdf/output'
batch_size = 32
lr_rate = 1e-3
total_epoch = 100

train_dataset = PartnetMobilityDataset(list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
)))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3
)

encoder.train()
decoder.train()

losses = []

print('training on device = ', device)

for epoch in range(total_epoch):
    for batch, (cp, sp, occ) in enumerate(train_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        print(batch)

        c = encoder(cp)
        p = decoder(c, sp)

        logits = decoder(p, c)
        loss_0 = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')

        optimizer.zero_grad()
        loss_0.backward()
        optimizer.step()

        losses.append(loss_0.item())

        print('[epoch %d, batch %d] loss: %.3f' % (epoch, batch, loss_0.item()))

print(losses)