from rich import print
import torch
from transformer.utils import to_cuda
from tqdm import tqdm
from torch import distributions
import os


class Trainer():
    def __init__(self, wandb_instance, config, model, dataloader, n_epoch,
                 ckpt_save_name, betas, eps, scheduler_factor, scheduler_warmup):
        self.n_epoch = n_epoch
        self.ckpt_save_name = ckpt_save_name
        self.device = config['device']
        self.input_structure = config['part_structure']
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.d_model = config['model_parameter']['d_model']
        self.wandb_instance = wandb_instance
        self.called = False
        self.g_token0_z = distributions.Normal(torch.zeros(self.d_model, device=self.device),
                                               torch.ones(self.d_model, device=self.device))

        # lr: not important, will be overriden by the scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1, betas=betas, eps=float(eps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                              lr_lambda=lambda step:
                                                                  self.rate(step, self.d_model, scheduler_factor, scheduler_warmup))


    # @from: https://nlp.seas.harvard.edu/annotated-transformer/#batches-and-masking
    @classmethod
    def rate(cls, step, model_size, factor, warmup):
        """
        we have to default the step to 1 for LambdaLR function
        to avoid zero raising to negative power.
        """
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def compute_loss(self, index, input, output):
        predicted_shape, g_token_dist = self.model(index, input)

        loss_kl = distributions.kl_divergence(g_token_dist, self.g_token0_z).sum(dim=-1).mean()
        # print('loss_kl', loss_kl)

        keys = list(self.input_structure['non_latent_info'].keys())

        loss_pred = torch.zeros(1, device=self.device)
        for key in keys:
            loss_pred += torch.nn.functional.mse_loss(predicted_shape[key], output[key])
        # print('loss_pred', loss_pred)
        loss = loss_pred + loss_kl
        return loss

    def save_checkpoint(self, epoch):
        ckptpath = self.ckpt_save_name % epoch
        os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
        torch.save(self.model, ckptpath)

    def feed_to_wandb(self, args):
        if self.wandb_instance:
            self.wandb_instance.log(args)

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch in range(self.n_epoch):
            self.model.train()

            train_losses = []
            for idx, (d_idx, input, output) in tqdm(enumerate(self.dataloader), desc=f'epoch = {epoch}', total=len(self.dataloader)):
                # print(f'idx = {idx}')
                if self.device == 'cuda':
                    (d_idx, input, output) = to_cuda((d_idx, input, output))

                loss = self.compute_loss(d_idx, input, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch =', epoch, 'loss =', torch.tensor(train_losses).mean(), 'lr =', lr)

            self.feed_to_wandb({
                'train_loss': torch.tensor(train_losses).mean(),
                'lr': self.optimizer.param_groups[0]['lr']
            })


