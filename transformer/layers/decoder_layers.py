from torch import nn
from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionWiseFeedForward
from transformer.layers.norm import LayerNorm
from transformer.utils import parse_args

class NativeDecoderLayer(nn.Module):
    def __init__(self, config, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.self_attention = MultiHeadAttention(n_head, d_model)

        self.ffn = PositionWiseFeedForward(**parse_args(config, config['feedforward']['args']))

        self.dropout_0 = nn.Dropout(dropout)
        self.norm_0 = LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = LayerNorm(d_model)

    def forward(self, x, mask):
        before_x = x
        x = self.self_attention(x, x, x, mask)

        x = self.dropout_0(x)
        x = self.norm_0(x + before_x)

        before_x = x
        x = self.ffn(x)

        x = self.dropout_1(x)
        x = self.norm_1(x + before_x)

        return x