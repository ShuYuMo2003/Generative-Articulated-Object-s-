import torch
import torch.nn as nn
import torch.nn.functional as F
from onet.fourier_feature import GaussianFourierFeatureEmbedding


# Max Pooling operation
def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension if output code z
        # c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, dim=3, emb_sigma=12.):
        super().__init__()
        self.z_dim = z_dim

        # Submodules
        self.fourier_feature = GaussianFourierFeatureEmbedding(d_emb=z_dim * 2, d_in=dim, emb_sigma=emb_sigma)
        self.fc_pos_1 = nn.Linear(z_dim * 2, z_dim)

        self.fc_0 = nn.Linear(1, z_dim)
        self.fc_1 = nn.Linear(z_dim, z_dim)
        self.fc_2 = nn.Linear(z_dim * 2, z_dim)
        self.fc_3 = nn.Linear(z_dim * 2, z_dim)
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_logstd = nn.Linear(z_dim, z_dim)

        self.actvn = nn.GELU()
        self.pool = maxpool


    def forward(self, p, x, c=None, **kwargs):
        batch_size, T, D = p.size()

        # output size: B x T X F
        p = self.fourier_feature(p)
        p = self.fc_pos_1(p)
        x = self.fc_0(x.unsqueeze(-1))
        net = x + p

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        # Reduce
        #  to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
