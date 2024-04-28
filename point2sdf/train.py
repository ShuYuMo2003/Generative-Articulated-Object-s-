import torch
from torch.utils.data import DataLoader
from torch import distributions
from torch.nn import functional as F
from glob import glob
from pathlib import Path

from decoder import Decoder
from pointnet_encoder import SimplePointnet as Encoder
from dataset import PartnetMobilityDataset

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

dataset_root_path = '/home/shuyumo/research/GAO/point2sdf/output'
batch_size = 8
lr_rate = 1e-2
total_epoch = 1000

train_dataset = PartnetMobilityDataset(list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
)))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    batch_loss = []
    for batch, (cp, sp, occ) in enumerate(train_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        c = encoder(cp)
        logits = decoder(c, sp)

        loss_0 = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        loss = loss_0.sum(-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())

    print(f'epoch {epoch} loss = {torch.tensor(batch_loss).mean()}')
    writer.add_scalar("Loss/train", torch.tensor(batch_loss).mean(), epoch)
    writer.flush()

writer.close()