import os
import torch
import wandb

import numpy as np
from rich import print
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from transformer.utils import to_cuda
from transformer.model import get_decoder
from transformer.loaddataset import get_dataset

from utils.evaluators import GenSDFLatentCodeEvaluator
from utils.logging import Log, console

torch.autograd.set_detect_anomaly(True)

class Trainer():
    def __init__(self, wandb_instance, config, n_epoch,
                 ckpt_save_path, betas, eps, scheduler_factor, per_epoch_save,
                 scheduler_warmup, latent_code_loss_ratio, sample_image_freq, vq_loss_ratio):

        self.n_epoch = n_epoch
        self.per_epoch_save = per_epoch_save

        self.ckpt_save_path = Path(ckpt_save_path) / datetime.now().strftime(r'%m-%d-%H-%M-%S')
        self.ckpt_save_path.mkdir(parents=True, exist_ok=True)


        self.device = config['device']
        self.input_structure = config['part_structure']
        self.model = get_decoder(config).to(self.device)
        self.sample_image_freq = sample_image_freq
        self.vq_loss_ratio = vq_loss_ratio

        assert config['decoder']['type'] == 'DecoderV2'

        self.train_dataset = get_dataset(config, train=True)
        self.train_dataloader = DataLoader(self.train_dataset, **config['dataloader']['args'])

        self.evaluate_dataset = get_dataset(config, train=False)
        Log.info('evaluate_dataset = %s.', len(self.evaluate_dataset))
        self.evaluate_dataloader = DataLoader(self.evaluate_dataset, **config['evaluate_dataloader']['args'])

        self.d_model = config['model_parameter']['d_model']
        self.wandb_instance = wandb_instance
        self.called = False

        self.latent_code_loss_ratio = latent_code_loss_ratio
        assert 0 <= self.latent_code_loss_ratio <= 1, 'latent_code_loss_ratio should be in [0, 1]'

        e_config = config['evaluator']
        self.latent_code_evaluator = GenSDFLatentCodeEvaluator(
            gensdf_model_path=Path(self.train_dataset.get_onet_ckpt_path()),
            eval_mesh_output_path=Path(e_config['eval_mesh_output_path']),
            resolution=e_config['resolution'],
            max_batch=e_config['max_batch'],
            device=self.device
        )

        # lr: not important, will be overriden by the scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1, betas=betas, eps=float(eps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            lr_lambda=lambda step:
                                                            self.rate(step, self.d_model,
                                                                      scheduler_factor, scheduler_warmup))

    def zero_gred(self):
        self.optimizer.zero_grad()

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

    def compute_loss(self, batched_data):
        input, output, padding_mask, output_skip_end_token_mask, enc_data, enc_data_raw = batched_data
        vq_loss, predicted_output = self.model(input, padding_mask, enc_data)

        dim_latent_code = self.input_structure['latent_code']
        # batch * part_idx * d_model
        predicted_latent_code = predicted_output[:, :, -dim_latent_code:] * padding_mask.unsqueeze(-1)
        predicted_other_info  = predicted_output[:, :, :-dim_latent_code] * padding_mask.unsqueeze(-1)

        latent_code = output[:, :, -dim_latent_code:] * padding_mask.unsqueeze(-1)
        other_info  = output[:, :, :-dim_latent_code] * padding_mask.unsqueeze(-1)

        loss_latent = torch.nn.functional.mse_loss(predicted_latent_code, latent_code)
        loss_other  = torch.nn.functional.mse_loss(predicted_other_info, other_info)

        loss = self.latent_code_loss_ratio * loss_latent + (1 - self.latent_code_loss_ratio) * loss_other

        loss = loss + self.vq_loss_ratio * vq_loss

        return loss, loss_latent, loss_other, vq_loss

    def save_checkpoint(self, epoch):
        Log.info('Save checkpoint at epoch %s', epoch)
        ckpt_name = self.ckpt_save_path / f'ckpt_{epoch}.pth'
        torch.save(self.model, ckpt_name.as_posix())
        Log.info('Done, Saved checkpoint at epoch %s, %s', epoch, ckpt_name)

    def feed_to_wandb(self, args):
        Log.info('%s', args)
        if self.wandb_instance:
            self.wandb_instance.log(args)

    def run_epoch(self, epoch_idx):
        self.model.train()
        train_losses = {
            'loss': [],
            'loss_latent': [],
            'loss_pred': [],
            'loss_vq': [],
        }
        for idx, batched_data in tqdm(enumerate(self.train_dataloader),
                                                desc=f'epoch = {epoch_idx}',
                                                total=len(self.train_dataloader)):

            if self.device == 'cuda':
                batched_data = to_cuda(batched_data)

            loss, loss_latent, loss_pred, vq_loss = self.compute_loss(batched_data)

            self.zero_gred()
            loss.backward()
            self.optimizer.step()
            train_losses['loss'].append(loss.item())
            train_losses['loss_latent'].append(loss_latent.item())
            train_losses['loss_pred'].append(loss_pred.item())
            train_losses['loss_vq'].append(vq_loss.item())
            # print({
            #         'loss': loss.item(),
            #         'loss_latent': loss_latent.item(),
            #         'loss_pred': loss_pred.item(),
            #     })
        return train_losses

    def evaluate_shape_acc(self):
        self.model.eval()

        expected_output = []
        predicted_output = []
        valid_output_mask = []

        with torch.no_grad():
            for idx, batched_data in tqdm(enumerate(self.evaluate_dataloader),
                                                    desc='Evaluating',
                                                    total=len(self.evaluate_dataloader)):

                if self.device == 'cuda':
                    batched_data = to_cuda(batched_data)

                input, output, padding_mask, output_skip_end_token_mask,  \
                            encoded_text, text = batched_data

                vq_loss, p_output = self.model(input, padding_mask, encoded_text)

                expected_output.append(output)
                predicted_output.append(p_output)
                valid_output_mask.append((padding_mask == 1) & (output_skip_end_token_mask == 1))

        expected_output = torch.cat(expected_output, dim=0)
        predicted_output = torch.cat(predicted_output, dim=0)
        valid_output_mask = torch.cat(valid_output_mask, dim=0)

        dim_latent_code = self.input_structure['latent_code']

        expected_latentcode_output = expected_output[:, :, -dim_latent_code:]
        predicted_latentcode_output = predicted_output[:, :, -dim_latent_code:]

        pred_images = self.latent_code_evaluator.screenshoot(predicted_latentcode_output, valid_output_mask, 5)
        gt_images = self.latent_code_evaluator.screenshoot(expected_latentcode_output, valid_output_mask, 5)
        images = []

        for pred_img, gt_img in zip(pred_images, gt_images):
            image = np.concatenate([gt_img, pred_img], axis=0)
            images.append(image)

        total_image = np.concatenate(images, axis=1)

        return total_image

    def __call__(self):
        assert not self.called, 'Trainer can only be called once'
        self.called = True

        for epoch_idx in range(self.n_epoch):

            train_losses = self.run_epoch(epoch_idx)

            self.lr_scheduler.step()

            if epoch_idx != 0 and epoch_idx % self.per_epoch_save == 0:
                self.save_checkpoint(epoch_idx)

            log_data = {
                'train_loss': torch.tensor(train_losses['loss']).mean(),
                'train_loss_latent': torch.tensor(train_losses['loss_latent']).mean(),
                'train_loss_pred': torch.tensor(train_losses['loss_pred']).mean(),
                'train_loss_vq': torch.tensor(train_losses['loss_vq']).mean(),
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            if epoch_idx != 0 and epoch_idx % self.sample_image_freq == 0:
                log_data['shape_image'] = wandb.Image(self.evaluate_shape_acc(), caption='shape image')
            self.feed_to_wandb(log_data)