import torch
from tqdm import tqdm, trange
import pyvista as pv
from onet_v2.utils.generate_3d import Generator3D
from pathlib import Path

class LatentCodeEvaluator:
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
        self.generator = Generator3D(device=device, threshold=0.4, resolution0=20)


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



