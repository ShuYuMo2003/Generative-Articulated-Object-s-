import torch
from torch import nn
from transformer.layers.decoder_layers import NativeDecoderLayer
from transformer.embedding import get_tokenizer
from transformer.embedding.position import NativeCatPositionEmbedding
from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.embedding.untokenizer import UnTokenizer
from transformer.embedding.g_embedding import GTokenEmbedding

class NativeDecoder(nn.Module):
    def __init__(self, config, n_head, d_model, vocab_size, n_layer, max_len, expanded_d_model, dropout, latent_code_dim):
        super().__init__()
        self.device = config['device']
        self.tokenizer = get_tokenizer(config, d_model=d_model, latent_code_dim=latent_code_dim)
        self.position_embedding = NativeCatPositionEmbedding(d_model=d_model, max_len=max_len, device=config['device'])
        self.layers = nn.ModuleList([NativeDecoderLayer(config, n_head=n_head, d_model=d_model, dropout=dropout) for _ in range(n_layer)])
        self.g_token_embedder = GTokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.untokenizer = UnTokenizer(d_model, expanded_d_model, latent_code_dim, dropout)

    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device)
        mask = torch.tril(mask)
        return mask

    def forward(self, index, raw_parts, mask=None):
        tokens = self.tokenizer(raw_parts)

        g_token_dist = self.g_token_embedder(index) # batch * d_model

        g_token_sample = g_token_dist.rsample()

        tokens = torch.cat((g_token_sample, tokens), dim=1)

        # n_batch, n_part, d_model
        tokens = self.position_embedding(tokens)

        n_batch, n_part, d_model = tokens.size()

        mask = self.generate_mask(n_part)

        for layer in self.layers:
            tokens = layer(tokens, mask)

        p_token = tokens[:, 0, :]
        part_info = self.untokenizer(p_token)

        return part_info, g_token_sample


