import torch
from torch import nn

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        self.encoding = torch.zeros(max_len, d_model, requires_grad=False, device=device)

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)

        _2i = torch.arange(0, d_model, 2, device=device).float().unsqueeze(0)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

class NativeCatPositionEmbedding(SinusoidalPositionEmbedding):
    def __init__(self, d_model, max_len, device):
        super().__init__(d_model, max_len, device)
        self.combine = nn.Linear(2 * d_model, d_model)


    def forward(self, tokens):
        batch, seq_len, d_model = tokens.size()
        embedding = self.encoding[:seq_len, :]
        embedding[-1, :] *= 2
        tokens = nn.cat(tokens, self.encoding[:seq_len, :], dim=-1)

        tokens = self.combine(tokens)
        return tokens

class NativeAddtionPositionEmbedding(SinusoidalPositionEmbedding):
    def __init__(self, d_model, max_len, device):
        super().__init__(d_model, max_len, device)

    def forward(self, tokens):
        batch, seq_len, d_model = tokens.size()
        tokens = tokens + self.encoding[:seq_len, :]
        return tokens