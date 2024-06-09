
import os
import copy
import torch

from torch import nn
from rich import print
import pyvista as pv
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from transformer.loaddataset.redis_parallel import identity_or_create_tensor

from transformer.loaddataset import get_dataset
from transformer.utils import to_cuda
from onet.generate_3d import Generator3D
from onet.decoder import Decoder


class Evaluater():
    def __init__(self, config, ckpt_filepath, eval_output_path, equal_part_threshold):
        self.model = torch.load(ckpt_filepath)

        self.dataset = get_dataset(config)

        self.latent_decoder = torch.load(self.dataset.onet_decoder_path)

        self.config = config
        self.device = config['device']

        self.eval_output_path = eval_output_path
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.s_part = copy.deepcopy(self.dataset.s_part)
        self.e_part = copy.deepcopy(self.dataset.e_part)

        if self.device == 'cuda':
            self.s_part = to_cuda(self.s_part)
            self.e_part = to_cuda(self.e_part)

        self.equal_part_threshold = equal_part_threshold

        self.basic_shape_structure = {}
        self.basic_shape_structure.update(config['part_structure']['non_latent_info'])
        self.basic_shape_structure.update(config['part_structure']['latent_info'])

        self.generator = Generator3D(device=self.device)

    def compare_last_part_with_e(self, total_part):
        result = 0
        for key in self.basic_shape_structure.keys():
            result += nn.functional.mse_loss(total_part[key][0, -1, :], self.e_part[key])
        print('mse loss', result)
        return result

    def add_s_part_n_add_pos(self, part_list):
        # print('part_list', part_list)
        # print('self.s_part', self.s_part)

        for key in self.basic_shape_structure.keys():
            if part_list.get(key, None) is not None:
                part_list[key] = torch.cat((self.s_part[key].unsqueeze(0).unsqueeze(0), part_list[key]), dim=1)
            else:
                part_list[key] = self.s_part[key].unsqueeze(0).unsqueeze(0)

        current_length = part_list[key].size(1)

        part_list['dfn'] = torch.tensor([i  for i in range(current_length)], device=self.device).unsqueeze(0)
        part_list['dfn_fa'] = torch.tensor([0 if i == 0 else i-1  for i in range(current_length)], device=self.device).unsqueeze(0)
        # print('after part_list', part_list)
        return part_list

    def generate_mesh_from_latent(self, latent):
        """
        latent: (1, latent_dim)
        """
        assert latent.size(0) == 1, latent.dim() == 2
        with torch.no_grad():
            mesh = self.generator.generate_from_latent(self.latent_decoder, latent)
        return mesh

    def generate_screenshoot_from_mesh(self, mesh):
        """
        mesh: Trimesh.mesh
        """
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

    def visualize(self, total_part):
        latent = total_part['latent']
        meshs = []
        for i in range(latent.size(1)):
            mesh = self.generate_mesh_from_latent(latent[:, i, :])

            screenshot = self.generate_screenshoot_from_mesh(mesh)
            plt.imshow(screenshot)
            plt.savefig(os.path.join(self.eval_output_path, f"{i}.png"))

            meshs.append(mesh)

    def inference(self):
        self.model.eval()
        count = 0
        key_padding_mask = torch.ones((1, self.dataset.fix_length), device=self.device, dtype=torch.int16)
        total_part = self.add_s_part_n_add_pos({})
        with torch.no_grad():
            while True:
                # print(total_part)
                total_part, _ = self.model(None, total_part, key_padding_mask[:, :total_part['dfn'].size(1)])
                count += 1
                total_part = self.add_s_part_n_add_pos(total_part)
                if self.compare_last_part_with_e(total_part) < self.equal_part_threshold or count >= 3: break

        self.visualize(total_part)
        print(total_part)



