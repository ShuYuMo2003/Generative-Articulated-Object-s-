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
    def __init__(self, input_structure, d_model, hidden_dim, latent_code_dim, leaky_relu, drop_out):
        super().__init__()
        self.structure      = input_structure
        self.part_info_dim  = sum(self.structure['non_latent_info'].values())

        self.part_info_fc   = nn.Linear(self.part_info_dim, hidden_dim)
        self.latent_code_fc = nn.Linear(latent_code_dim, hidden_dim)

        self.combine_fc     = nn.Linear(hidden_dim * 2, d_model)
        self.activ          = nn.LeakyReLU(leaky_relu) if leaky_relu else nn.ReLU()
        self.dropout        = nn.Dropout(drop_out)

    def forward(self, raw_parts):
        # raw_parts: attribute_name * batch * (part_idx==fix_length) * attribute_dim
        # print(raw_parts)
        part_tensor     = torch.cat([raw_parts[key] for key, value in self.structure['non_latent_info'].items()], dim=-1)
        part_tensor     = self.part_info_fc(part_tensor)

        latent_tensor   = self.latent_code_fc(raw_parts['latent'])

        x = torch.cat((part_tensor, latent_tensor), dim=-1)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.combine_fc(x)       # x: batch * part_idx * d_model

        dfn = raw_parts['dfn']       # batch * (part_idx==fix_length)
        dfn_fa = raw_parts['dfn_fa'] # batch * (part_idx==fix_length)

        return dfn, dfn_fa, x