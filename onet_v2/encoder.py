import torch
import torch.nn as nn
import torch.nn.functional as F
from onet_v2.fourier_feature import GaussianFourierFeatureEmbedding


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
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, emb_sigma=12.):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        self.fourier_feature = GaussianFourierFeatureEmbedding(d_emb=256, d_in=dim, emb_sigma=emb_sigma)
        self.fc_pos_0 = nn.Linear(256, 256)
        self.pos_acti = nn.ReLU()
        self.fc_pos_1 = nn.Linear(256, 128)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, 128)

        self.fc_0 = nn.Linear(1, 128)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

        # if not leaky:
        #     self.actvn = F.relu
        #     self.pool = maxpool
        # else:
        #     self.actvn = F.leaky_relu
        #     self.pool = torch.mean

    def forward(self, p, x, c=None, **kwargs):
        batch_size, T, D = p.size()

        # output size: B x T X F
        p = self.fourier_feature(p)
        # p = self.fc_pos_0(p)
        # p = self.pos_acti(p)
        p = self.fc_pos_1(p)
        x = self.fc_0(x.unsqueeze(-1))
        net = x + p

        if self.c_dim != 0:
            net = net + self.fc_c(c).unsqueeze(1)

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
