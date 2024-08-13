import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from onet.decoder import Decoder
from onet.encoder import Encoder

class ONet(nn.Module):
    def __init__(self, dim_z, emb_sigma):
        super().__init__()

        self.encoder = Encoder(z_dim=dim_z, emb_sigma=emb_sigma)
        self.decoder = Decoder(z_dim=dim_z, emb_sigma=emb_sigma, hidden_size=dim_z)
        self.dim_z = dim_z

    def get_decoder(self):
        return self.decoder

    def forward(self, enc_sp, enc_occ, dec_sp):
        mean_z, logstd_z = self.encoder(enc_sp, enc_occ)
        q_z = distributions.Normal(mean_z, torch.exp(logstd_z))
        z = q_z.rsample()
        p0_z = distributions.Normal(
            torch.zeros(self.dim_z, device=mean_z.device),
            torch.ones(self.dim_z, device=mean_z.device)
        )
        kl = distributions.kl_divergence(q_z, p0_z).sum(dim=-1)
        kl_loss = kl.mean()

        logits_or_sdf = self.decoder(dec_sp, z)

        return logits_or_sdf, kl_loss, mean_z
