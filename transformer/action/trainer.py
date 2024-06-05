from rich import print
import torch
from transformer.utils import to_cuda
from tqdm import tqdm
from torch import distributions
from onet.decoder import Decoder as LatentDecoder
from torch.utils.data import DataLoader
from transformer.model import get_decoder
from transformer.loaddataset import get_dataset
import os


class Trainer():
    def __init__(self, wandb_instance, config, n_epoch,
                 ckpt_save_name, betas, eps, scheduler_factor, per_epoch_save,
                 scheduler_warmup, evaluetaion, n_validation_sample, main_loss_ratio):

        self.n_epoch = n_epoch
        self.per_epoch_save = per_epoch_save
        self.ckpt_save_name = ckpt_save_name
        self.device = config['device']
        self.input_structure = config['part_structure']
        self.model = get_decoder(config).to(self.device)

        self.model_type = config['decoder']['type']
        if self.model_type == 'NativeDecoder':
            self.compute_loss = self.compute_loss_g_token
        elif self.model_type == 'ParallelDecoder':
            self.compute_loss = self.compute_loss_parallel

        self.train_dataset = get_dataset(config)
        self.validate_dataset = get_dataset(config, train=False)
        self.train_dataloader = DataLoader(self.train_dataset, **config['dataloader']['args'])

        self.d_model = config['model_parameter']['d_model']
        self.wandb_instance = wandb_instance
        self.called = False
        self.g_token0_z = distributions.Normal(torch.zeros(self.d_model, device=self.device),
                                               torch.ones(self.d_model, device=self.device))
        self.main_loss_ratio = main_loss_ratio
        assert 0 <= self.main_loss_ratio <= 1, 'main_loss_ratio should be in [0, 1]'

        # lr: not important, will be overriden by the scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1, betas=betas, eps=float(eps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            lr_lambda=lambda step:
                                                            self.rate(step, self.d_model,
                                                                      scheduler_factor, scheduler_warmup))
        self.run_latent_eval = evaluetaion
        if evaluetaion:
            self.latent_decoder = torch.load(self.train_dataset.onet_decoder_path).to(self.device)
            self.latent_decoder.eval()

            self.n_latent_code_validation_samples = config['n_latent_code_validation_samples']
            self.n_validation_sample = n_validation_sample


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

        batch_size, n_part, latent_dim = pred_latent.shape
        assert pred_latent.shape == actu_latent.shape,        \
            'pred_latent and actu_latent should have the same shape'
        sampled_point = torch.rand((self.n_latent_code_validation_samples, 3),
                                   device=self.device) - 0.5

        sampled_point = sampled_point.unsqueeze(0).repeat(batch_size, 1, 1)

        # only take the first part for validation.
        pred_latent = pred_latent[:, 0, :]
        actu_latent = actu_latent[:, 0, :]

        with torch.no_grad():
            pred_occ_logits = self.latent_decoder(sampled_point, pred_latent)
            actu_occ_logits = self.latent_decoder(sampled_point, actu_latent)

        acc = ((pred_occ_logits > 0) == (actu_occ_logits > 0)).float().mean()

        return acc

    def run_valiate_shape_acc(self):
        validate_dataloader = DataLoader(self.validate_dataset, batch_size=1, num_workers=1)
        acc = []
        for idx, (d_idx, input, output) in tqdm(enumerate(validate_dataloader),
                                                desc='validate shape quality.',
                                                total=self.n_validation_sample):
            if idx >= self.n_validation_sample:
                break

            if self.device == 'cuda':
                (d_idx, input, output) = to_cuda((d_idx, input, output))

            predicted_shape, _ = self.model(d_idx, input)
            # attribute_name * n_batch * (part_idx==fix_length) * attribute_dim
            shape_acc = self.validate_shape_acc(predicted_shape['latent'], output['latent'])
            acc.append(shape_acc)

        return torch.tensor(acc).mean()

    def compute_loss_g_token(self, index, input, output):
        predicted_shape, g_token_dist = self.model(index, input)

        loss_kl = distributions.kl_divergence(g_token_dist, self.g_token0_z).sum(dim=-1).mean()

        keys = list(self.input_structure['non_latent_info'].keys())

        loss_pred = torch.zeros(1, device=self.device)
        for key in keys:
            loss_pred += torch.nn.functional.mse_loss(predicted_shape[key], output[key])

        loss = loss_pred + loss_kl
        return loss, loss_pred, loss_kl

    def compute_loss_parallel(self, index, input, output):
        predicted_shape, _ = self.model(index, input)

        keys = list(self.input_structure['non_latent_info'].keys())

        loss_pred = torch.zeros(1, device=self.device)
        for key in keys:
            loss_pred += torch.nn.functional.mse_loss(predicted_shape[key], output[key])
        return loss_pred

    def save_checkpoint(self, epoch):
        print('save checkpoint at epoch', epoch, '...', end='')
        ckptpath = self.ckpt_save_name.format(epoch=epoch)
        os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
        torch.save(self.model, ckptpath)
        print('done')

    def feed_to_wandb(self, args):
        print(args)
        if self.wandb_instance:
            self.wandb_instance.log(args)

    def __call__(self):

        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch_idx in range(self.n_epoch):
            self.model.train()

            train_losses = []
            for idx, (d_idx, input, output) in tqdm(enumerate(self.train_dataloader),
                                                    desc=f'epoch = {epoch_idx}',
                                                    total=len(self.train_dataloader)):
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

            self.feed_to_wandb({
                'train_loss': torch.tensor(train_losses).mean(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'shape_acc': shape_acc
            })


