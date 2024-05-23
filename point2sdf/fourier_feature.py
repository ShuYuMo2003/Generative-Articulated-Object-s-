import torch
from torch import nn


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_or_generate_fourier_matrix_B(d_out_emb, d_in_emb):
    seed_torch(hash('ytq') & ((1 << 25) - 1))
    cache_file = f'fourier_marix_cache/fourier_matrix_B_{d_out_emb}_{d_in_emb}.pth'
    try:
        B = torch.load(cache_file)
    except FileNotFoundError:
        B = torch.randn(d_out_emb, d_in_emb)
        torch.save(B, cache_file)
    return B

class GaussianFourierFeatureEmbedding(nn.Module):
    def __init__(self, d_emb=256, d_in=3, emb_sigma=12., device='cuda'):
        super().__init__()
        assert d_emb % 2 == 0, 'd_emb must be even. (half sin, half cos)'
        self.B = get_or_generate_fourier_matrix_B(d_emb // 2, d_in) * emb_sigma
        self.B = self.B.to(device)
        self.B.requires_grad_ = False

    def to(self, device):
        super().to(device)
        self.B = self.B.to(device)
        return self

    def forward(self, x):
        # (....... * d_in) ----> (....... * d_emb)
        return torch.cat((torch.sin(2 * torch.pi * x @ self.B.t()),
                          torch.cos(2 * torch.pi * x @ self.B.t())), dim=-1)



