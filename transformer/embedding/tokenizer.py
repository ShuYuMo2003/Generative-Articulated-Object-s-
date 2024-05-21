import torch
from rich import print
from torch import nn

'''
example:
'dfn': 1,
'dfn_fa': 0,
'bounds': array([-0.03526 ,  0.522128, -0.15423 ,  0.159007, -0.709672,  0.923116], dtype=float32),
'tran': array([0., 0., 0.], dtype=float32),
'direction': array([0., 0., 0.], dtype=float32),
'origin': array([0., 0., 0.], dtype=float32),
'limit': array([0., 0., 0., 0.], dtype=float32),
'latent': array([-1.21896137e-02,  2.56319121e-02, -5.50454366e-04, -3.20652053e-02,
'''


class NativeMLPTokenizer(nn.Module):
    def __init__(self, d_model, hidden_dim, latent_code_dim, drop_out, leaky_relu=None):
        super().__init__()
        self.origin_fc      = nn.Linear(3, hidden_dim)
        self.direction_fc   = nn.Linear(3, hidden_dim)
        self.bounds_fc      = nn.Linear(6, hidden_dim)
        self.trans_fc       = nn.Linear(3, hidden_dim)
        self.limit_fc       = nn.Linear(4, hidden_dim)
        self.latent_code_fc = nn.Linear(latent_code_dim, hidden_dim)
        self.combine_fc     = nn.Linear(hidden_dim * 6, d_model)
        self.activ          = nn.LeakyReLU(leaky_relu) if leaky_relu else nn.ReLU()
        self.dropout       = nn.Dropout(drop_out)

    def forward(self, raw_parts):
        # raw_parts: attribute_name * batch * (part_idx==fix_length) * attribute_dim
        # print(raw_parts)
        tokenized_parts = []
        # print(raw_parts['origin'].device)
        part_tensor = torch.cat((
            self.origin_fc(raw_parts['origin']),
            self.direction_fc(raw_parts['direction']),
            self.bounds_fc(raw_parts['bounds']),
            self.trans_fc(raw_parts['tran']),
            self.limit_fc(raw_parts['limit']),
            self.latent_code_fc(raw_parts['latent'].float()),)
        , dim=-1)
        # print(part_tensor.shape)
        x = self.activ(part_tensor)
        x = self.dropout(x)
        tokenized_parts_latent = self.combine_fc(x)
        # tokenized_parts_latent: batch * part_idx * d_model
        # print('tokenized_parts_latent: ', tokenized_parts_latent.shape)
        dfn = raw_parts['dfn']
        dfn_fa = raw_parts['dfn_fa']
        # print('dfn   :', dfn.shape)    # batch * (part_idx==fix_length)
        # print('dfn_fa:', dfn_fa.shape) # batch * (part_idx==fix_length)
        return dfn, dfn_fa, tokenized_parts_latent