import torch
import numpy as np
import random
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
train_ratio = 0.9

dataset_path = list(zip(
    glob(dataset_root_path + '/pointcloud/*'),
    glob(dataset_root_path + '/point/*')
))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(encoder, decoder, val_dataloader):
    losses = []
    accurancys = []
    with torch.no_grad():
        for batch, (cp, sp, occ) in enumerate(val_dataloader):
            cp = cp.to(device)
            sp = sp.to(device)
            occ = occ.to(device)

            c = encoder(cp)
            logits = decoder(c, sp)

            loss_0 = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
            loss = loss_0.sum(-1).mean()

            accurancy = (logits > 0).float().eq(occ).float().mean()

            losses.append(loss)
            accurancys.append(accurancy)
    return {
        'loss': torch.tensor(losses).mean(),
        'acc': torch.tensor(accurancys).mean()
    }



setup_seed(hash('ytq') & ((1 << 32) - 1))

train_dataset = PartnetMobilityDataset(dataset_path,
                                       train_ratio=train_ratio, train=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)

val_dataset = PartnetMobilityDataset(dataset_path,
                                       train_ratio=train_ratio, train=False)
val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=True)

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
print('train dataset length = ', len(train_dataset))
print('valda dataset length = ', len(val_dataset))

for epoch in range(total_epoch):
    batch_loss = []
    batch_acc = []
    for batch, (cp, sp, occ) in enumerate(train_dataloader):
        cp = cp.to(device)
        sp = sp.to(device)
        occ = occ.to(device)

        c = encoder(cp)
        logits = decoder(c, sp)

        acc = (logits > 0).float().eq(occ).float().mean()

        loss_0 = F.binary_cross_entropy_with_logits(logits, occ, reduction='none')
        loss = loss_0.sum(-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    validation_dict = validate(encoder, decoder, val_dataloader)
    writer.add_scalar("Loss/val", validation_dict['loss'], epoch)
    writer.add_scalar("Acc/val", validation_dict['acc'], epoch)
    writer.add_scalar("Loss/train", torch.tensor(batch_loss).mean(), epoch)
    writer.add_scalar("Acc/train", torch.tensor(batch_acc).mean(), epoch)
    writer.flush()

    print(f'epoch {epoch} loss = {torch.tensor(batch_loss).mean()}')


writer.close()