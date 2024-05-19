from torch import nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.acti = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.acti(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
