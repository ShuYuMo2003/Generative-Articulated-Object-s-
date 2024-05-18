from torch import nn

class NativeDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

    def forward(self, x, mask):
        pass