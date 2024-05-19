from torch import nn
from math import sqrt

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        n_batch, n_head, n_part, d_model = q.size()

        k_t = k.permute(0, 1, 3, 2)

        score = (q @ k_t) / sqrt(d_model)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attention = self.softmax(score)

        v = attention @ v

        return v