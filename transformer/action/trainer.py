from rich import print
import torch
from transformer.utils.utils import to_cuda
from tqdm import tqdm


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
        predicted_shape = self.model(input)
        n_batch = predicted_shape['origin'].size(0)
        keys = ['origin', 'direction', 'bounds', 'tran', 'limit', 'latent']
        loss = torch.zeros(1, device=self.device)
        for key in keys:
            loss += torch.nn.functional.mse_loss(predicted_shape[key], output[key])
        return loss

    def feed_to_wandb(self, args):
        if self.wandb_instance:
            self.wandb_instance.log(args)

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch in range(self.n_epoch):
            self.model.train()

            train_losses = []
            for idx, (input, output) in tqdm(enumerate(self.dataloader), desc=f'epoch = {epoch}', total=len(self.dataloader)):
                # print(f'idx = {idx}')
                if self.device == 'cuda':
                    (input, output) = to_cuda((input, output))

                loss = self.compute_loss(input, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            print(f'epoch {epoch} loss = {torch.tensor(train_losses).mean()}')

            self.feed_to_wandb({
                'train_loss': torch.tensor(train_losses).mean()
            })


