import os
import torch
import wandb
import pyvista as pv
import numpy   as np

from rich import print
from tqdm import tqdm
from torch import distributions
from torch.nn import functional as F

from onet.decoder import Decoder as LatentDecoder
from onet.generate_3d import Generator3D

from transformer.utils import to_cuda
from torch.utils.data import DataLoader
from transformer.model import get_decoder
from transformer.loaddataset import get_dataset

torch.autograd.set_detect_anomaly(True)

class Trainer():
    def __init__(self, wandb_instance, config, n_epoch, gen_visualization_per_epoch,
                 ckpt_save_name, betas, eps, scheduler_factor, per_epoch_save,
                 scheduler_warmup, evaluetaion, n_validation_sample, latent_code_loss_ratio):

        self.n_epoch = n_epoch
        self.per_epoch_save = per_epoch_save
        self.ckpt_save_name = ckpt_save_name
        self.device = config['device']
        self.input_structure = config['part_structure']
        self.model = get_decoder(config).to(self.device)
        self.gen_visualization_per_epoch = gen_visualization_per_epoch

        self.model_type = config['decoder']['type']
        if self.model_type == 'NativeDecoder':
            self.compute_loss = self.compute_loss_g_token
        elif self.model_type == 'ParallelDecoder':
            self.compute_loss = self.compute_loss_parallel
        elif self.model_type == 'DecoderV2':
            self.compute_loss = self.compute_loss_v2

        self.train_dataset = get_dataset(config)
        # self.validate_dataset = get_dataset(config, train=False)
        self.train_dataloader = DataLoader(self.train_dataset, **config['dataloader']['args'])

        self.d_model = config['model_parameter']['d_model']
        self.wandb_instance = wandb_instance
        self.called = False

        # g token is deprecated
        self.g_token0_z = distributions.Normal(torch.zeros(self.d_model, device=self.device),
                                               torch.ones(self.d_model, device=self.device))


        self.latent_code_loss_ratio = latent_code_loss_ratio
        assert 0 <= self.latent_code_loss_ratio <= 1, 'latent_code_loss_ratio should be in [0, 1]'

        # lr: not important, will be overriden by the scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1, betas=betas, eps=float(eps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            lr_lambda=lambda step:
                                                            self.rate(step, self.d_model,
                                                                      scheduler_factor, scheduler_warmup))
        self.run_latent_eval = evaluetaion
        if evaluetaion:
            self.latent_decoder = torch.load(self.train_dataset.onet_decoder_path, map_location=torch.device(self.device)).to(self.device)

            # freeze latent decoder, we don't want to train it in this stage, just for clear gradient.
            # self.decoder_optimizer = torch.optim.Adam(self.latent_decoder.parameters(), lr=1, betas=betas, eps=float(eps))
            self.n_latent_code_validation_samples = config['n_latent_code_validation_samples']
            self.n_validation_sample = n_validation_sample

        self.generator       = Generator3D(device=self.device)

    def zero_gred(self):
        self.optimizer.zero_grad()
        # if self.run_latent_eval:
        #     self.decoder_optimizer.zero_grad()

    def gen_image_from_latent(self, decoder, mean_z):
        with torch.no_grad():
            # print('gen latent mesh')
            mesh = self.generator.generate_from_latent(decoder, mean_z)
            # print('gened')
            mesh.export('logs/temp-validate.obj')
            plotter = pv.Plotter()
            try:
                pv_mesh = pv.read('logs/temp-validate.obj')
                plotter.add_mesh(pv_mesh)
            except ValueError:
                print('error')
                pass
            plotter.show()
            screenshot = plotter.screenshot()
            return screenshot

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

    def run_valiate_shape_acc(self, with_image=False):
        validate_dataloader = DataLoader(self.validate_dataset, batch_size=1, num_workers=1, shuffle=False)
        acc = []
        actually_shape_list = []
        predicted_shape_list = []
        text_list = []
        for idx, batched_data in tqdm(enumerate(validate_dataloader),
                                                desc='validate shape quality.',
                                                total=self.n_validation_sample):
            if idx >= self.n_validation_sample:
                break

            if self.device == 'cuda':
                batched_data = to_cuda(batched_data)

            predicted_shape, _ = self.model(index           = batched_data['index'],
                                            raw_parts       = batched_data['input'],
                                            key_padding_mask= batched_data['key_padding_mask'],
                                            enc_data        = batched_data['enc_data'])

            # attribute_name * n_batch * (part_idx==fix_length) * attribute_dim
            shape_acc = self.validate_shape_acc(predicted_shape['latent'], batched_data['output']['latent'])
            actually_shape = batched_data['output']
            acc.append(shape_acc)
            actually_shape_list.append(actually_shape)
            predicted_shape_list.append(predicted_shape)
            text_list.append(batched_data['enc_text'])

        # batch_size, n_part, latent_dim
        images = None
        if with_image:
            images = []
            for actually_shape, predicted_shape in tqdm(zip(actually_shape_list, predicted_shape_list), desc='generate image'):
                single_obj_images = []
                for idx in range(3):
                    latent = predicted_shape['latent'][[0], idx, :]
                    img0 = self.gen_image_from_latent(self.latent_decoder, latent)
                    latent = actually_shape['latent'][[0], idx, :]
                    img1 = self.gen_image_from_latent(self.latent_decoder, latent)
                    img = np.concatenate([img0, img1], axis=1)
                    single_obj_images.append(img)
                single_obj_image = np.concatenate(single_obj_images, axis=0)
                images.append(single_obj_image)
            images = np.concatenate(images, axis=1)
            print('texts: ', text_list)


        return torch.tensor(acc).mean(), images

    def compute_loss_g_token(self, index, input, output, key_padding_mask):
        raise NotImplementedError('compute_loss_g_token is deprecated.')
        predicted_shape, g_token_dist = self.model(index, input)

        loss_kl = distributions.kl_divergence(g_token_dist, self.g_token0_z).sum(dim=-1).mean()

        keys = list(self.input_structure['non_latent_info'].keys()) + list(self.input_structure['latent_info'].keys())

        loss_pred = torch.zeros(1, device=self.device)
        for key in keys:
            loss_pred += torch.nn.functional.mse_loss(predicted_shape[key], output[key])

        loss = loss_pred + loss_kl
        return loss, loss_pred, loss_kl

    def compute_loss_parallel(self, index, input, output, key_padding_mask, enc_data, enc_text):
        predicted_shape, _ = self.model(index, input, key_padding_mask, enc_data)

        keys = list(self.input_structure['non_latent_info'].keys()) + list(self.input_structure['latent_info'].keys())

        loss_pred = torch.zeros(1, device=self.device)
        loss_latent = torch.zeros(1, device=self.device)

        for key in keys:
            if key == 'latent':
                loss_latent += torch.nn.functional.mse_loss(predicted_shape[key], output[key])
            else:
                loss_pred += torch.nn.functional.mse_loss(predicted_shape[key], output[key])

        return loss_latent + loss_pred, loss_latent, loss_pred

    def compute_loss_v2(self, batched_data):
        input, output, padding_mask, encoded_text, text = batched_data
        predicted_output = self.model(input, padding_mask, encoded_text)

        dim_latent_code = self.input_structure['latent_code']
        # batch * part_idx * d_model
        predicted_latent_code = predicted_output[:, :, -dim_latent_code:] * padding_mask.unsqueeze(-1)
        predicted_other_info  = predicted_output[:, :, :-dim_latent_code] * padding_mask.unsqueeze(-1)

        latent_code = output[:, :, -dim_latent_code:] * padding_mask.unsqueeze(-1)
        other_info  = output[:, :, :-dim_latent_code] * padding_mask.unsqueeze(-1)

        loss_latent = torch.nn.functional.mse_loss(predicted_latent_code, latent_code)
        loss_other  = torch.nn.functional.mse_loss(predicted_other_info, other_info)

        loss = self.latent_code_loss_ratio * loss_latent + (1 - self.latent_code_loss_ratio) * loss_other

        return loss, loss_latent, loss_other

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

            train_losses = {
                'loss': [],
                'loss_latent': [],
                'loss_pred': [],
            }
            for idx, batched_data in tqdm(enumerate(self.train_dataloader),
                                                    desc=f'epoch = {epoch_idx}',
                                                    total=len(self.train_dataloader)):

                if self.device == 'cuda':
                    batched_data = to_cuda(batched_data)

                loss, loss_latent, loss_pred = self.compute_loss(batched_data)

                self.zero_gred()
                loss.backward()
                self.optimizer.step()
                train_losses['loss'].append(loss.item())
                train_losses['loss_latent'].append(loss_latent.item())
                train_losses['loss_pred'].append(loss_pred.item())
                print({
                        'loss': loss.item(),
                        'loss_latent': loss_latent.item(),
                        'loss_pred': loss_pred.item(),
                    })
                self.lr_scheduler.step()

            # shape_acc, img = self.run_valiate_shape_acc(epoch_idx % self.gen_visualization_per_epoch == 0)

            if epoch_idx != 0 and epoch_idx % self.per_epoch_save == 0:
                self.save_checkpoint(epoch_idx)



            log_data = {
                'train_loss': torch.tensor(train_losses['loss']).mean(),
                'train_loss_latent': torch.tensor(train_losses['loss_latent']).mean(),
                'train_loss_pred': torch.tensor(train_losses['loss_pred']).mean(),
                'lr': self.optimizer.param_groups[0]['lr'],
                # 'shape_acc': shape_acc,
            }
            # if img is not None:
                # log_data['img'] = wandb.Image(img, caption='validation image: val[0]')
            self.feed_to_wandb(log_data)