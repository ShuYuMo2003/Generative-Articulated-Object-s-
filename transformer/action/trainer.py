from rich import print
import torch
from transformer.utils import to_cuda
from tqdm import tqdm
from torch import distributions
from point2sdf.decoder import Decoder as LatentDecoder
from torch.utils.data import DataLoader
import os


class Trainer():
    def __init__(self, validate_dataset, wandb_instance, config, model, dataloader, n_epoch,
                 ckpt_save_name, betas, eps, scheduler_factor, per_epoch_save,
                 scheduler_warmup, evaluetaion):
        self.validate_dataset = validate_dataset
        self.n_epoch = n_epoch
        self.per_epoch_save = per_epoch_save
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
        self.run_latent_eval = evaluetaion
        if evaluetaion:
            self.latent_decoder = LatentDecoder(**config['latent_decoder']['args'])
            self.latent_decoder.load_state_dict(torch.load(config['latent_decoder']['ckpt_path']))
            self.latent_decoder = self.latent_decoder.to(self.device)
            self.latent_decoder.eval()
            self.n_latent_code_validation_samples = config['latent_decoder']['n_latent_code_validation_samples']


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

    def validate_shape_acc(self, pred_latent, actu_latent):
        self.latent_decoder.eval()
        batch_size, latent_dim = pred_latent.shape
        assert (batch_size, latent_dim) == actu_latent.shape, 'pred_latent and actu_latent should have the same shape'
        sampled_point = torch.rand((self.n_latent_code_validation_samples, 3)) - 0.5

        with torch.no_grad():
            pred_occ_logits = self.latent_decoder(pred_latent, sampled_point)
            actu_occ_logits = self.latent_decoder(actu_latent, sampled_point)

        acc = ((pred_occ_logits > 0) == (actu_occ_logits > 0)).float().mean()

        return acc

    def run_valiate_shape_acc(self):
        validate_dataloader = DataLoader(self.validate_dataset, batch_size=1, num_workers=1)
        acc = []
        for idx, (d_idx, input, output) in enumerate(validate_dataloader):
            if self.device == 'cuda':
                (d_idx, input, output) = to_cuda((d_idx, input, output))

            predicted_shape, _ = self.model(d_idx, input)
            shape_acc = self.validate_shape_acc(predicted_shape['latent'], output['latent'])
            acc.append(shape_acc)
            if idx > 10: # TODO: write in config
                break
        return torch.tensor(acc).mean()

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
        print('save checkpoint at epoch', epoch, '...', end='')
        ckptpath = self.ckpt_save_name.format(epoch=epoch)
        os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
        torch.save(self.model, ckptpath)
        print('done')

    def feed_to_wandb(self, args):
        if self.wandb_instance:
            self.wandb_instance.log(args)

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch_idx in range(self.n_epoch):
            self.model.train()

            train_losses = []
            for idx, (d_idx, input, output) in tqdm(enumerate(self.dataloader), desc=f'epoch = {epoch_idx}', total=len(self.dataloader)):
                # print(f'idx = {idx}')
                if self.device == 'cuda':
                    (d_idx, input, output) = to_cuda((d_idx, input, output))

                loss = self.compute_loss(d_idx, input, output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            shape_acc = self.run_valiate_shape_acc()

            if epoch_idx != 0 and epoch_idx % self.per_epoch_save == 0:
                self.save_checkpoint(epoch_idx)

            self.lr_scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch =', epoch_idx, 'loss =', torch.tensor(train_losses).mean(), 'lr =', lr)

            self.feed_to_wandb({
                'train_loss': torch.tensor(train_losses).mean(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'shape_acc': shape_acc
            })


