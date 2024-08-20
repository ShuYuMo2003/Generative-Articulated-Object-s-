import torch

from utils.logging import Log, console
from torch import nn
from torch.nn import functional as F

from einops import reduce

# Partially copied from GenSDF and Diffusion-SDF.

from .autoencoder import BetaVAE
from .sdf_model import SdfModel

class SDFModulationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sdf_model = SdfModel(config)

        self.config = config
        feature_dim = config["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet
        modulation_dim = feature_dim*3                      # latent dim of during modulation
        latent_std = config["latent_std"]                   # std of target gaussian distribution of latent space
        latent_dim = config["latent_dim"]                   # latent dim of modulation
        hidden_dims = [modulation_dim, modulation_dim,
                       modulation_dim, modulation_dim, modulation_dim ]
        self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=latent_dim, hidden_dims=hidden_dims, kl_std=latent_std)

    def forward(self, x, current_epoch):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.config["kld_weight"] )
        except Exception as e:
            Log.critical("vae loss is nan at epoch %s... message: %s", current_epoch, e)
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        # loss_dict =  {"sdf": sdf_loss, "vae": vae_loss}
        # self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss, reconstructed_plane_feature