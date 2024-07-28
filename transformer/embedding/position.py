import torch
from transformer.layers.layernormGRU import LayerNormGRUCell
from torch import nn

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, requires_grad=False, device=device)

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)

        _2i = torch.arange(0, d_model, 2, device=device).float().unsqueeze(0)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

class PositionGRUEmbedding(nn.Module):
    def __init__(self, d_model, dim_single_emb, dropout):
        super().__init__()
        self.dim_single_emb = dim_single_emb

        self.fc0_comp = nn.Linear(d_model, d_model)
        self.dropout_comp = nn.Dropout(dropout)
        self.act_comp = nn.GELU()
        self.fc1_comp = nn.Linear(d_model, dim_single_emb)

        self.combine_fc = nn.Linear(2 * d_model, d_model)

        self.gru = LayerNormGRUCell(d_model, d_model)

    def mlp_compress(self, x):
        x = self.fc0_comp(x)
        x = self.act_comp(x)
        x = self.dropout_comp(x)
        x = self.fc1_comp(x)
        return x

    def forward(self, tokens):
        # ('token'/'fa') * batch * part_idx * attribute_dim
        # tokens
        batch, seq_len, d_model = tokens['token'].size()
        gru_emb = torch.zeros((batch, seq_len, d_model), device=tokens['token'].device)
        fa = tokens['fa']
        for part_idx in range(seq_len):
            # batch, d_model
            hx = gru_emb[torch.arange(batch), fa[:, part_idx], :]
            x = tokens['token'][:, part_idx, :]
            gru_emb[:, part_idx, :] = self.gru(x, hx)

        shorted_gru_emb = self.mlp_compress(gru_emb)

        emb = torch.zeros((batch, seq_len, d_model), device=tokens['token'].device)
        for part_idx in range(seq_len):
            prev = emb[torch.arange(batch), fa[:, part_idx], :-self.dim_single_emb]
            current = shorted_gru_emb[:, part_idx, :]
            emb[:, part_idx, :] = torch.cat((current, prev), dim=-1)

        # print('3 emb', emb.shape)
        # torch.set_printoptions(threshold=2000000)
        # with open('logs/debug/emb0.txt', 'w') as f:
        #     f.write(emb[0].__str__())

        # with open('logs/debug/emb1.txt', 'w') as f:
        #     f.write(emb[1].__str__())

        # with open('logs/debug/emb2.txt', 'w') as f:
        #     f.write(emb[2].__str__())


        tokens = self.combine_fc(torch.cat((tokens['token'], emb), dim=-1))

        return tokens

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
        embedded_tokenized_parts_list = torch.cat((position_embedding, tokenized_parts_list), dim=-1)

        embedded_tokenized_parts_list = self.combine(embedded_tokenized_parts_list)

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