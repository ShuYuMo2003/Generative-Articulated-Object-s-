from torch import nn

class MLPUnTokenizerV2(nn.Module):
    def __init__(self, d_token, d_hidden, d_model, drop_out):
        super().__init__()
        self.fc_0 = nn.Linear(d_model, d_hidden)
        self.acti = nn.Mish()
        self.drop = nn.Dropout(drop_out)
        self.fc_1 = nn.Linear(d_hidden, d_token)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.acti(x)
        x = self.drop(x)
        x = self.fc_1(x)
        return x

# Deprecated
class NativeMLPUnTokenizer(nn.Module):
    def __init__(self, input_structure, d_model, expanded_d_model, latent_code_dim, dropout):
        super().__init__()
        self.structure      = input_structure
        self.part_info_dim  = sum(self.structure['non_latent_info'].values())

        self.expand_layer   = nn.Linear(d_model, expanded_d_model)
        self.latent_code_fc = nn.Linear(expanded_d_model, latent_code_dim)
        self.part_info_fc   = nn.Linear(expanded_d_model, self.part_info_dim)

        self.activ = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # n_batch, n_part, d_model
        x = self.expand_layer(tokens)
        x = self.activ(x)
        x = self.dropout(x)

        latent = self.latent_code_fc(x)
        part_info = self.part_info_fc(x)

        result = {}
        d = 0
        for key, dim in self.structure['non_latent_info'].items():
            result[key] = part_info[..., d:d+dim]
            d += dim
        # attribute_name, n_batch, n_part * attribute_dim
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


