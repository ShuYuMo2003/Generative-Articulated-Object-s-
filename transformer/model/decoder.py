import torch
from torch import nn
from rich import print
from transformer.utils import parse_args
from transformer.layers.decoder_layers import NativeDecoderLayer
from transformer.embedding import (get_tokenizer,   get_positionembedding,
                                   get_g_embedding, get_untokenizer)

class NativeDecoder(nn.Module):
    def __init__(self, config, n_layer, device):
        super().__init__()
        self.device         = device
        self.tokenizer      = get_tokenizer(config)
        self.position_emb   = get_positionembedding(config)
        self.g_token_emb    = get_g_embedding(config)
        self.untokenizer    = get_untokenizer(config)
        self.layers         = nn.ModuleList([
            NativeDecoderLayer(config, **parse_args(config, config['decoder']['layer_arges']))
                for _ in range(n_layer)
        ])

    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device)
        mask = torch.tril(mask)
        return mask

    def forward(self, index, raw_parts, mask=None):
        dfn, dfn_fa, tokens = self.tokenizer(raw_parts)

        g_token_dist = self.g_token_emb(index) # batch * d_model

        # print(type(g_token_dist), g_token_dist)
        g_token_sample = g_token_dist.rsample()
        # print('dddd', g_token_sample)
        # print('g_token_sample', g_token_sample.shape)
        # print('tokens', tokens.shape)

        # Replace the first token in each batch with g_token.
        # Initially, the first token in each batch is a zero tensor which is defined in `redis.py`.
        # I only need the `dfn`/`dfn_fa` for g_token determined by dataset.
        tokens[:, 0, :] = g_token_sample

        # n_batch, n_part, d_model
        tokens = self.position_emb((dfn, dfn_fa, tokens))

        n_batch, n_part, d_model = tokens.size()

        mask = self.generate_mask(n_part)

        for layer in self.layers:
            tokens = layer(tokens, mask)

        p_token = tokens[:, 0, :]

        part_info = self.untokenizer(p_token)

        return part_info, g_token_dist


class ParallelDecoder(nn.Module):
    def __init__(self, config, n_layer, device):
        super().__init__()
        self.device         = device
        self.tokenizer      = get_tokenizer(config)
        self.position_emb   = get_positionembedding(config)
        self.untokenizer    = get_untokenizer(config)
        self.layers         = nn.ModuleList([
            NativeDecoderLayer(config, **parse_args(config, config['decoder']['layer_arges']))
                for _ in range(n_layer)
        ])

    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device)
        mask = torch.tril(mask)
        return mask

    def forward(self, index, raw_parts, mask=None):
        dfn, dfn_fa, tokens = self.tokenizer(raw_parts)

        # print(dfn.shape, dfn_fa.shape, tokens.shape)

        # n_batch, n_part, d_model
        tokens = self.position_emb((dfn, dfn_fa, tokens))

        n_batch, n_part, d_model = tokens.size()

        mask = self.generate_mask(n_part)

        for layer in self.layers:
            tokens = layer(tokens, mask)

        part_info = self.untokenizer(tokens)

        return part_info, None