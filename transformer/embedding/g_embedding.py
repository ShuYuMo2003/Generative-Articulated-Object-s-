import torch
from torch import nn
from torch import distributions

class NativeGEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding_mean = nn.Embedding(vocab_size, d_model)
        self.embedding_logstd = nn.Embedding(vocab_size, d_model)

    def forward(self, idx):
        return distributions.Normal(self.embedding_mean(idx),
                                    torch.exp(self.embedding_logstd(idx)))

# class DirectGEmbedding(nn.Module):
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)

#     def forward(self, idx):
#         return distributions.Normal(self.embedding_mean(idx),
#                                     torch.exp(self.embedding_logstd(idx)))
