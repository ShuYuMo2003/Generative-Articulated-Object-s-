from torch import nn
from transformer.layers.decoder_layers import NativeDecoderLayer
from transformer.embedding import get_tokenizer
from transformer.embedding.position import NativeCatPositionEmbedding

class NativeDecoder(nn.Module):
    def __init__(self, config, n_head, d_model, n_layer, max_len):
        super().__init__()
        self.device = config['device']
        self.tokenizer = get_tokenizer(config)
        self.position_embedding = NativeCatPositionEmbedding(d_model, max_len=max_len, device=config['device'])
        self.layers = nn.ModuleList([NativeDecoderLayer(n_head, d_model) for _ in range(n_layer)])

    def forward(self, raw_parts, mask=None):
        total_parts = self.tokenizer(raw_parts)
        tokens = self.position_embedding(total_parts)

        # print(parent, '\n\n', part_latent)
        # for layer in self.layers:
        #     x = layer(x, mask)
        # return x

