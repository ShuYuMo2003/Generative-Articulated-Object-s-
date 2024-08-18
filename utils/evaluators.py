import torch
import trimesh

from utils.logging import Log, console
from tqdm import tqdm, trange
import pyvista as pv
from onet.utils.generator_sim import Generator3DSimple
from pathlib import Path
from gensdf.utils import mesh as MeshUtils
from gensdf.utils import generate_mesh_screenshot

class GenSDFLatentCodeEvaluator:
    def __init__(self, gensdf_model_path: Path, eval_mesh_output_path: Path,
                 resolution: int, max_batch: int, device: str):
        self.device = device
        self.gensdf = torch.load(gensdf_model_path.as_posix())
        self.gensdf.eval()
        self.gensdf = self.gensdf.to(device)
        self.eval_mesh_output_path = eval_mesh_output_path
        eval_mesh_output_path.mkdir(exist_ok=True, parents=True)
        self.resolution = resolution
        self.max_batch = max_batch

        Log.info('GenSDFLatentCodeEvaluator Set Up with model_path = %s', gensdf_model_path)
        Log.info('Resolution = %d', resolution)
        Log.info('Max Batch = %d', max_batch)

    def generate_mesh(self, z: torch.Tensor):
        with torch.no_grad():
            mesh = self.generator.generate_from_latent(self.onet.decoder, z)
        return mesh

    def screenshoot(self, z: torch.Tensor, mask: torch.Tensor, cut_off: int):
        z = z.reshape(-1, z.size(-1))
        mask = mask.reshape(-1)
        z = z[mask][:cut_off]

        images = []

        for batch in tqdm(range(z.shape[0]),
                          desc="Generating Mesh"):
            recon_latent = z[[batch]]
            output_mesh = (self.eval_mesh_output_path / f'GenSDFLatentCodeEvaluator_{batch}.ply').as_posix()
            MeshUtils.create_mesh(self.gensdf.sdf_model, recon_latent,
                            output_mesh, N=self.resolution,
                            max_batch=self.max_batch,
                            from_plane_features=True)
            mesh = trimesh.load(output_mesh)
            # Log.debug('Loaded Mesh %s', mesh)
            screenshot = generate_mesh_screenshot(mesh)
            images.append(screenshot)

        return images

class OnetLatentCodeEvaluator:
    def __init__(self, onet_model_path: Path,
                       n_sample_point: int,
                       onet_batch_size: int,
                       device: str):
        self.device = device
        self.n_sample_point = n_sample_point
        self.onet = torch.load(onet_model_path.as_posix())
        self.onet.eval()
        self.onet = self.onet.to(device)
        self.onet_batch_size = onet_batch_size
        self.generator = Generator3DSimple(device=device, threshold=0.4, resolution0=20)


    def get_accuracy(self, generated_z: torch.Tensor, z: torch.Tensor,
                        mask: torch.Tensor):
        '''
            z: latent code with n_batch * seq_len * dim_z
        '''
        generated_z = generated_z.reshape(-1, generated_z.size(-1))
        z = z.reshape(-1, z.size(-1))
        mask = mask.reshape(-1)

        generated_z = generated_z[mask]
        z = z[mask]

        total_size = generated_z.size(0)

        assert total_size == z.size(0), "Unconsistent with `generated_z` and `z`."

        sampled_point = torch.rand((self.n_sample_point, 3), device=self.device) - 0.5

        generated_logits = []
        logits = []
        with torch.no_grad():
            for i in trange(0, total_size, self.onet_batch_size):
                next_i = min(i+self.onet_batch_size, total_size)
                current_batch_size = next_i - i

                batched_sampled_point = sampled_point.unsqueeze(0).expand(current_batch_size, -1, -1)
                batched_generated_z = generated_z[i:next_i]
                batched_z = z[i:next_i]

                generated_logit = self.onet.decoder(batched_sampled_point, batched_generated_z)
                logit = self.onet.decoder(batched_sampled_point, batched_z)

                generated_logits.append(generated_logit)
                logits.append(logit)

        generated_logits = torch.cat(generated_logits, dim=0)
        logits = torch.cat(logits, dim=0)

        acc = ((generated_logits > 0) == (logits > 0)).float().mean()

        return acc

    def generate_mesh(self, z: torch.Tensor):
        with torch.no_grad():
            mesh = self.generator.generate_from_latent(self.onet.decoder, z)
        return mesh

    def screenshoot(self, z: torch.Tensor, mask: torch.Tensor, cut_off: int):
        z = z.reshape(-1, z.size(-1))
        mask = mask.reshape(-1)
        z = z[mask][:cut_off]

        images = []

        for batch in tqdm(z, desc="Gen Meshs"):
            mesh = self.generate_mesh(batch.unsqueeze(0))
            plotter = pv.Plotter(off_screen=True)
            if mesh.faces.size != 0:
                plotter.add_mesh(mesh)
            plotter.show()
            screenshot = plotter.screenshot()
            images.append(screenshot)

        return images



