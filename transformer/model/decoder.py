from torch import nn
from transformer.layers.decoder_layers import NativeDecoderLayer


class NativeDecoder(nn.Module):
    def __init__(self, n_head, d_model, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([NativeDecoderLayer(n_head, d_model) for _ in range(n_layers)])

