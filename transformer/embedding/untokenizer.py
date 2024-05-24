from torch import nn


class NativeMLPUnTokenizer(nn.Module):
    def __init__(self, input_structure, d_model, expanded_d_model, latent_code_dim, dropout):
        super().__init__()
        self.structure      = input_structure
        self.part_info_dim  = sum(self.structure['non_latent_info'].values())

        self.expand_layer   = nn.Linear(d_model, expanded_d_model)
        self.latent_code_fc = nn.Linear(expanded_d_model, latent_code_dim)
        self.part_info_fc   = nn.Linear(expanded_d_model, self.part_info_dim)

        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, p_token):
        x = self.expand_layer(p_token)
        x = self.activ(x)
        x = self.dropout(x)

        latent = self.latent_code_fc(x)
        part_info = self.part_info_fc(x)

        result = {}
        d = 0
        for key, dim in self.structure['non_latent_info'].items():
            result[key] = part_info[..., d:d+dim]
            d += dim
        assert d == part_info.size(-1)

        result['latent'] = latent
        return result

        # return {
        #     'origin': self.origin_fc(expanded_p_token),
        #     'direction': self.direction_fc(expanded_p_token),
        #     'bounds': self.bounds_fc(expanded_p_token),
        #     'tran': self.trans_fc(expanded_p_token),
        #     'limit': self.limit_fc(expanded_p_token),
        #     'latent': self.latent_code_fc(expanded_p_token)
        # }


