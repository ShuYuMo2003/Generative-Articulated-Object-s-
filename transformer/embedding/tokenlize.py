from torch import nn

class NativeMLPTokenizer(nn.Module):
    def __init__(self, d_model, hidden_dim, leaky_relu):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(leaky_relu) if leaky_relu else nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, tokens):
        batch, seq_len, d_model = tokens.size()
        tokens = self.mlp(tokens)
        return tokens