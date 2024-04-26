from torch import nn

from decoder import Decoder
from pointnet_encoder import SimplePointnet as Encoder
from torch import distributions

class OccupancyNetwork(nn.Module):
    def __init__(self, c_dim):
        self.encoder = Encoder(c_dim=c_dim)
        self.decoder = Decoder(c_dim=c_dim, leaky=1e-2)

    def forward(self, cloud_points, sample_points):
        latent_c    = self.encoder(cloud_points)
        logits      = self.decoder(latent_c, sample_points)
        occupancy   = distributions.Bernoulli(logits=logits)

        return occupancy

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)