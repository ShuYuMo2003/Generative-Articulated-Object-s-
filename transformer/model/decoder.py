import torch
from torch import nn
from rich import print

from transformer.utils import parse_args
from transformer.layers.decoder_layers import NativeDecoderLayer
from transformer.embedding import (get_tokenizer,   get_positionembedding,
                                   get_g_embedding, get_untokenizer)
from transformer.layers.post_encoder import PostEncoder

from transformer.layers.quantizer import VectorQuantizer

class DecoderV2(nn.Module):
    def __init__(self, config, n_layer, device):
        super().__init__()
        self.device         = device
        self.d_model        = config['model_parameter']['d_model']
        self.tokenizer      = get_tokenizer(config)
        self.position_emb   = get_positionembedding(config)
        self.untokenizer    = get_untokenizer(config)
        self.postencoder    = PostEncoder(dim=config['model_parameter']['encoder_kv_dim'],
                                          deepth=config['model_parameter']['post_encoder_deepth'])

        vq_config = config['model_parameter']['vectorQuantizer']
        self.vq = VectorQuantizer(n_e=vq_config['n_e'], e_dim=vq_config['e_dim'],
                                  d_model=self.d_model, n_channel=vq_config['n_channel'],
                                  temperature=vq_config['temperature'], beta=vq_config['beta'],
                                  device = self.device)

        assert n_layer % 2 == 0, "n_layer must be even number"
        self.n_half_layer = n_layer // 2

        self.front_layers   = nn.ModuleList([
            NativeDecoderLayer(config, **parse_args(config, config['decoder']['layer_arges']))
                for _ in range(self.n_half_layer)
        ])
        self.rear_layers    = nn.ModuleList([
            NativeDecoderLayer(config, **parse_args(config, config['decoder']['layer_arges']))
                for _ in range(self.n_half_layer)
        ])

    def generate_mask(self, n_part):
        mask = torch.ones(n_part, n_part, device=self.device, dtype=torch.int16)
        # mask = torch.tril(mask) # no need mask
        return mask

    def forward(self, input, padding_mask, enc_data):
        # ('token'/'dfn'/'dfn_fa') * batch * part_idx * attribute_dim
        enc_data = self.postencoder(enc_data)

        batch, n_part, d_model = input['token'].size()

        # print('1 input[token]', input['token'].shape, 'input[fa]', input['fa'].shape)

        input['token'] = self.tokenizer(input['token'])
        tokens = self.position_emb(input)

        # print('2 tokens', tokens.shape)

        attn_mask = self.generate_mask(n_part)

        x_stack = []
        for idx, layer in enumerate(self.front_layers):
            tokens = layer(tokens, padding_mask, attn_mask, enc_data, None)
            x_stack.append(tokens)

        vq_loss, tokens = self.vq(tokens)

        for idx, layer in enumerate(self.rear_layers):
            tokens = layer(tokens, padding_mask, attn_mask, enc_data, x_stack.pop())

        tokens = self.untokenizer(tokens)

        return vq_loss, tokens