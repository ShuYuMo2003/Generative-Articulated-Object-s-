
import os
import torch
from rich import print
from torch.utils.data import DataLoader
from transformer.utils import to_cuda
from onet.generate_3d import Generator3D
from onet.decoder import Decoder

class LatentCodeParser():
    def __init__(self, config):
        self.device = config['device']
        self.generator = Generator3D(device=self.device)
        self.decoder = Decoder(**config['latent_decoder']['args'])
        checkpoint_state = torch.load(config['latent_decoder']['ckpt_path'])
        self.decoder.load_state_dict(checkpoint_state['decoder'])
        self.decoder = self.decoder.to(self.device)
        self.decoder.eval()

    def __call__(self, latent_code):
        # TODO: check whether batchsize can not be 1.
        mesh = self.generator.generate_from_latent(self.decoder, latent_code)
        return mesh

class Evaluater():
    def __init__(self, config, ckpt_path, dataset, eval_output):
        self.model = torch.load(ckpt_path)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.visual_generator = LatentCodeParser(config)
        self.config = config
        self.device = config['device']
        self.eval_output = eval_output
        os.makedirs(self.eval_output, exist_ok=True)

    def visualize_generated_shape(self):
        self.model.eval()
        for idx, (d_idx, input, output) in enumerate(self.dataloader):
            if self.device == 'cuda':
                (d_idx, input, output) = to_cuda((d_idx, input, output))

            # Not try to visualize E part. bad impl. TODO: add part type into returing dataset.
            if output['dfn'][0] == self.config['dataset']['args']['fix_length']: continue

            with torch.no_grad():
                predicted_shape, _ = self.model(d_idx, input)
                # print(idx, 'predpredpred', predicted_shape)
                # print(idx, 'oringoringir', output)
                mesh = self.visual_generator(predicted_shape['latent'])
                mesh.export(os.path.join(self.eval_output, f'{d_idx.item()}.obj'))
                print(d_idx, mesh, 'saved')