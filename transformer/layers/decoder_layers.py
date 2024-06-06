from torch import nn

from transformer.utils import parse_args
# from transformer.layers.norm import LayerNorm
# from transformer.attention.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import PositionWiseFeedForward



class NativeDecoderLayer(nn.Module):
    def __init__(self, config, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,
                                                    dropout=dropout, batch_first=True)

        self.ffn = PositionWiseFeedForward(**parse_args(config, config['feedforward']['args']))

        self.dropout_0 = nn.Dropout(dropout)
        self.norm_0 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask, attn_mask):
        before_x = x
        x, attn_weight = self.self_attention(x, x, x,
                                key_padding_mask=(key_padding_mask == 0),
                                attn_mask=(attn_mask == 0))

        x = self.dropout_0(x)
        x = self.norm_0(x + before_x)

        before_x = x
        x = self.ffn(x)

        x = self.dropout_1(x)
        x = self.norm_1(x + before_x)

        return x