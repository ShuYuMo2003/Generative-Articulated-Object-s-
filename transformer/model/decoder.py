import torch
from torch import nn
from transformer.layers.decoder_layers import NativeDecoderLayer
from transformer.embedding import get_tokenizer
from transformer.embedding.position import NativeCatPositionEmbedding
from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.embedding.untokenizer import UnTokenizer

class NativeDecoder(nn.Module):
    def __init__(self, config, n_head, d_model, n_layer, max_len, expanded_d_model, dropout, latent_code_dim):
        super().__init__()
        self.device = config['device']
        self.tokenizer = get_tokenizer(config, d_model=d_model, latent_code_dim=latent_code_dim)
        self.position_embedding = NativeCatPositionEmbedding(d_model=d_model, max_len=max_len, device=config['device'])
        self.layers = nn.ModuleList([NativeDecoderLayer(config, n_head=n_head, d_model=d_model, dropout=dropout) for _ in range(n_layer)])

        self.untokenizer = UnTokenizer(d_model, expanded_d_model, latent_code_dim, dropout)

    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device)
        mask = torch.tril(mask)
        return mask

    def forward(self, raw_parts, mask=None):
        tokenized_with_tree_info = self.tokenizer(raw_parts)

        # n_batch, n_part, d_model
        tokens = self.position_embedding(tokenized_with_tree_info)

        n_batch, n_part, d_model = tokens.size()

        mask = self.generate_mask(n_part)

        for layer in self.layers:
            tokens = layer(tokens, mask)

        p_token = tokens[:, 0, :]
        part_info = self.untokenizer(p_token)

        return part_info


