from torch import nn

from transformer.attention.single_attention_module import DotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.d_model        = d_model
        self.n_head         = n_head
        self.q_fc           = nn.Linear(d_model, d_model)
        self.k_fc           = nn.Linear(d_model, d_model)
        self.v_fc           = nn.Linear(d_model, d_model)
        self.combine_fc     = nn.Linear(d_model, d_model)
        self.attention      = DotProductAttention()

    def forward(self, q, k, v, mask=None):
        q, k, v =  self.q_fc(q),  self.k_fc(k),  self.v_fc(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out = self.attention(q, k, v, mask) # n_batch, n_head, n_part, d_model_per_head

        n_batch, n_head, n_part, d_model_per_head = out.size()
        d_model = d_model_per_head * n_head

        assert d_model == self.d_model, 'd_model should be divided by n_head'

        out = out.transpose(1, 2).contiguous().view(n_batch, n_part, d_model)
        out = self.combine_fc(out)

        return out


    def split(self, x):
        n_batch, n_part, d_model = x.size()
        assert d_model % self.n_head == 0, 'd_model should be divided by n_head'

        d_model_per_head = d_model // self.n_head
        x = x.view(n_batch, n_part, self.n_head, d_model_per_head).transpose(1, 2).contiguous()
        return x

