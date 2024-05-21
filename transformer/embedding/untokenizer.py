from torch import nn


class UnTokenizer(nn.Module):
    def __init__(self, d_model, expanded_d_model, latent_code_dim, dropout):
        super().__init__()
        self.expand_layer = nn.Linear(d_model, expanded_d_model)
        self.activ = nn.ReLU()
        self.origin_fc = nn.Linear(expanded_d_model, 3)
        self.direction_fc = nn.Linear(expanded_d_model, 3)
        self.bounds_fc = nn.Linear(expanded_d_model, 6)
        self.trans_fc  = nn.Linear(expanded_d_model, 3)
        self.limit_fc  = nn.Linear(expanded_d_model, 4)
        self.latent_code_fc = nn.Linear(expanded_d_model, latent_code_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, p_token):
        expanded_p_token = self.expand_layer(p_token)
        expanded_p_token = self.activ(expanded_p_token)
        return {
            'origin': self.origin_fc(expanded_p_token),
            'direction': self.direction_fc(expanded_p_token),
            'bounds': self.bounds_fc(expanded_p_token),
            'tran': self.trans_fc(expanded_p_token),
            'limit': self.limit_fc(expanded_p_token),
            'latent': self.latent_code_fc(expanded_p_token)
        }


