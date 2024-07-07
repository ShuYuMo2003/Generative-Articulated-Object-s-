
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

from transformers import AutoTokenizer, T5EncoderModel

import pyrender


class Evaluater():
    def __init__(self, config, ckpt_filepath, eval_output_path, equal_part_threshold):
        self.model = torch.load(ckpt_filepath, map_location=config['device'])

        self.dataset = get_dataset(config)

        self.latent_decoder = torch.load(self.dataset.onet_decoder_path, map_location=config['device'])

        self.config = config
        self.device = config['device']

        self.eval_output_path = eval_output_path
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.s_part = copy.deepcopy(self.dataset.s_part)
        self.e_part = copy.deepcopy(self.dataset.e_part)

        self.tokenizer = AutoTokenizer.from_pretrained(config['transformers_name'],
                                                       cache_dir=config['pretrained_model_cache_dir'])
        print('load tokenizer')
        self.text_encoder = T5EncoderModel.from_pretrained(config['transformers_name'],
                                                           cache_dir=config['pretrained_model_cache_dir'])
        print('load text encoder')

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
        # print('mse loss', result)
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

    def generate_box_bound(self, parts, meshs):
        pyrender_mesh = pyrender.Mesh.from_trimesh(meshs[0])
        scene = pyrender.Scene()
        scene.add(pyrender_mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)

    def genreate_each_mesh(self, total_part):
        latent = total_part['latent']
        meshs = []
        for i in range(latent.size(1) - 1): # do not genreate e_part
            mesh = self.generate_mesh_from_latent(latent[:, i, :])

            screenshot = self.generate_screenshoot_from_mesh(mesh)
            plt.imshow(screenshot)
            plt.savefig(os.path.join(self.eval_output_path, f"{i}.png"))
            print(os.path.join(self.eval_output_path, f"{i}.png"), 'generated')

            meshs.append(mesh)
        return meshs

    def inference(self, idx=None):
        self.model.eval()
        count = 0
        key_padding_mask = torch.ones((1, self.dataset.fix_length), device=self.device, dtype=torch.int16)
        total_part = self.add_s_part_n_add_pos({})
        if idx is not None:
            test_data_sample = self.dataset[idx]
            enc_data = identity_or_create_tensor(test_data_sample['enc_data']).to(self.device).unsqueeze(0)
            enc_text = test_data_sample['enc_text']
        else:
            enc_text = input("Input prompt for generation: ").strip()
            input_ids = self.tokenizer([enc_text], return_tensors="pt", padding=True).input_ids
            outputs = self.text_encoder(input_ids=input_ids)
            enc_data = outputs.last_hidden_state.to(self.device)

        print('runing with prompt:', enc_text)

        with torch.no_grad():
            while True:
                # print(total_part)
                total_part, _ = self.model(None, total_part, key_padding_mask[:, :total_part['dfn'].size(1)], enc_data)
                count += 1
                total_part = self.add_s_part_n_add_pos(total_part)
                mse_loss_with_e_part = self.compare_last_part_with_e(total_part)
                print('MSE Loss with E-part = ', mse_loss_with_e_part)
                if mse_loss_with_e_part < self.equal_part_threshold:
                    print('Same token with E-part, end inference.')
                    break

        meshs = self.genreate_each_mesh(total_part)

        self.generate_box_bound(total_part, meshs)
        # print(total_part)



