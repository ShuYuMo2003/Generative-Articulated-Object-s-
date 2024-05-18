from rich import print
import torch
from transformer.utils.utils import to_cuda


class Trainer():
    def __init__(self, wandb_instance, config, model, dataloader, n_epoch,
                 ckpt_save_name, lr, optimizer, device):
        self.n_epoch = n_epoch
        self.ckpt_save_name = ckpt_save_name
        self.lr = lr
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.wandb_instance = wandb_instance
        self.called = False
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.lr))
        else:
            raise NotImplemented(f'{optimizer}: optimizer is not supported')

    def compute_loss(self, batched_data):
        predicted = self.model(batched_data)

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch in range(self.n_epoch):
            self.model.train()
            for idx, batched_data in enumerate(self.dataloader):
                if self.device == 'cuda': batched_data = to_cuda(batched_data)

                loss = self.compute_loss(batched_data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


