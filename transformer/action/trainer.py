from rich import print
import torch
from transformer.utils.utils import to_cuda


class Trainer():
    def __init__(self, wandb_instance, config, model, dataloader, n_epoch,
                 ckpt_save_name, lr, optimizer):
        self.n_epoch = n_epoch
        self.ckpt_save_name = ckpt_save_name
        self.lr = lr
        self.device = config['device']
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.wandb_instance = wandb_instance
        self.called = False
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.lr))
        else:
            raise NotImplemented(f'{optimizer}: optimizer is not supported')

    def compute_loss(self, input, output):
        predicted = self.model(input)
        # TODO: loss between predicted and output

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch in range(self.n_epoch):
            self.model.train()
            for idx, (input, output) in enumerate(self.dataloader):
                if self.device == 'cuda':
                    (input, output) = to_cuda((input, output))

                loss = self.compute_loss(input, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


