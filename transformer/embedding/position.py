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
        # self.position_embedding_combine = nn.Linear(2 * d_model, d_model)
        self.combine = nn.Linear(2 * d_model, d_model)

    def forward(self, tokenized_with_tree_info):
        dfn, dfn_fa, tokenized_parts_latent = tokenized_with_tree_info

        n_batch, n_part, d_model = tokenized_parts_latent.size()

        tokenized_parts_list = tokenized_parts_latent.view(-1, d_model)
        dfn_list = dfn.view(-1)
        dfn_fa_list = dfn_fa.view(-1)

        dfn_embedding = self.encoding[dfn_list, :]
        dfn_fa_embedding = self.encoding[dfn_fa_list, :]

        position_embedding = dfn_embedding + dfn_fa_embedding
        embedded_tokenized_parts_list = torch.cat(position_embedding, tokenized_parts_list, dim=-1)

        embedded_tokenized_parts = embedded_tokenized_parts_list.view(n_batch, n_part, d_model)
        return embedded_tokenized_parts
        # self.encoding[:seq_len, :]

# class NativeAddtionPositionEmbedding(SinusoidalPositionEmbedding):
#     def __init__(self, d_model, max_len, device):
#         super().__init__(d_model, max_len, device)

#     def forward(self, tokens):
#         batch, seq_len, d_model = tokens.size()
#         tokens = tokens + self.encoding[:seq_len, :]
#         return tokens