import torch
from torch import nn

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, requires_grad=False, device=device)

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)

        _2i = torch.arange(0, d_model, 2, device=device).float().unsqueeze(0)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

class NativeCatPositionEmbedding(SinusoidalPositionEmbedding):
    def __init__(self, d_model, max_len, device):
        super().__init__(d_model, max_len, device)
        self.position_embedding_combine = nn.Linear(2 * d_model, d_model)
        self.combine = nn.Linear(2 * d_model, d_model)

    def forward(self, total_parts):
        # total_parts：
        # part_idx, 2(parent_idx / latent_code), batch

        n_batch = total_parts[0][0].size(0)
        n_part  = len(total_parts)

        parts = []

        for part_idx in range(n_part):
            parent_embedding   = self.encoding[total_parts[part_idx][0] + 1, :]
            current_embedding  = self.encoding[[part_idx + 1] * n_batch, :]
            position_embedding = self.position_embedding_combine(torch.cat((parent_embedding, current_embedding), dim=-1))
            total = position_embedding + total_parts[part_idx][1]
            print(total.shape)
            parts.append(total)

        # part_idx, batch, d_model
        embedded_tokens = torch.stack(parts, dim=0)

        # batch, part_idx, d_model
        embedded_tokens = embedded_tokens.permute(1, 0, 2).contiguous()

        return embedded_tokens

# class NativeAddtionPositionEmbedding(SinusoidalPositionEmbedding):
#     def __init__(self, d_model, max_len, device):
#         super().__init__(d_model, max_len, device)

#     def forward(self, tokens):
#         batch, seq_len, d_model = tokens.size()
#         tokens = tokens + self.encoding[:seq_len, :]
#         return tokens