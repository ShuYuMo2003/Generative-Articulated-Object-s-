import torch
from rich import print
from torch import nn



class NativeMLPTokenizer(nn.Module):
    def __init__(self, d_model, hidden_dim, latent_code_dim, leaky_relu=None):
        super().__init__()
        self.origin_fc      = nn.Linear(3, hidden_dim)
        self.direction_fc   = nn.Linear(3, hidden_dim)
        self.bounds_fc      = nn.Linear(6, hidden_dim)
        self.trans_fc       = nn.Linear(3, hidden_dim)
        self.latent_code_fc = nn.Linear(latent_code_dim, hidden_dim)
        self.combine_fc     = nn.Linear(hidden_dim * 5, d_model)
        self.activ          = nn.LeakyReLU(leaky_relu) if leaky_relu else nn.ReLU()


    def forward(self, raw_parts):
        # print(raw_parts)
        tokenized_parts = []
        for data_n_parent in raw_parts['part']:
            part_tensor = torch.cat((
                self.origin_fc(data_n_parent['origin']),
                self.direction_fc(data_n_parent['direction']),
                self.bounds_fc(data_n_parent['bounds']),
                self.trans_fc(data_n_parent['trans']),
                self.latent_code_fc(data_n_parent['latent'].float()))
            , dim=-1)
            tokenized_parts_latent = self.combine_fc(self.activ(part_tensor))
            tokenized_parts.append((data_n_parent['parent'], tokenized_parts_latent))
        # print(tokenized_parts)
        return tokenized_parts