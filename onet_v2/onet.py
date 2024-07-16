import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

from onet_v2.decoder import Decoder
from onet_v2.encoder import Encoder

class ONet(nn.Module):
    def __init__(self, dim_z, emb_sigma):
        super().__init__()

        self.encoder = Encoder(z_dim=dim_z, c_dim=0, emb_sigma=emb_sigma)
        self.decoder = Decoder(z_dim=dim_z, c_dim=0, emb_sigma=emb_sigma)
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

        logits = self.decoder(dec_sp, z)

        return logits, kl_loss, mean_z
        # loss_i = F.binary_cross_entropy_with_logits(
        #         logits, dec_occ, reduction='none')
        # loss = loss + loss_i.sum(-1).mean()

        # acc = ((logits > 0) == dec_occ).float().mean()

        # return loss, acc, mean_z
