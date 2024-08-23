import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# adapted from: https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
class VectorQuantizer(nn.Module):
    """
     Adapted from `Discretization bottleneck part of the VQ-VAE.`

        n_e: number of embeddings
        e_dim: dimension of embedding
        d_model: dimension of input
        n_channel: number of channels, input token will be expanded and split into `n_channel` with size of `e_dim`
        tempareture: temperature for softmax, used for sampling from codebook like LLM.
        beta: commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, d_model, n_channel, temperature, beta, device):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.n_channel = n_channel
        self.temperature = temperature

        self.device = device

        self.fc0 = nn.Linear(d_model, e_dim * n_channel)
        self.fc1 = nn.Linear(e_dim * n_channel, d_model)

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        """
        n_batch, n_seq, d_model = z.shape

        # n_batch * n_seq * d_model --> n_batch * n_seq * (n_channel * e_dim)
        expand_z = self.fc0(z)

        # n_batch * n_seq * (n_channel * e_dim) --> (n_batch * n_seq * n_channel) * e_dim
        channeled_z = expand_z.reshape(n_batch * n_seq * self.n_channel, self.e_dim)

        # (n_batch * n_seq * n_channel) * e_dim ---> (n_batch * n_seq * n_channel) * n_e
        logits_score = torch.sum(channeled_z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(channeled_z, self.embedding.weight.t())

        # print('logits_score', logits_score)

        logits_score = logits_score - torch.max(logits_score, dim=1, keepdim=True)[0]

        # see: https://github.com/huggingface/transformers/blob/d806fa3e92289876e01ab19c9e19e9264ea1c1a1/src/transformers/generation/utils.py#L2454
        # Do not find closest embedding.
        # Sample from codebook based on logis.
        prob = F.softmax((logits_score) / self.temperature, dim=1)
        min_encoding_indices = torch.multinomial(prob, num_samples=1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        channeled_z_q = torch.matmul(min_encodings, self.embedding.weight)

        # print('channeled_z_q', channeled_z_q)
        # print('self.embedding.weight', self.embedding.weight)

        # compute loss for embedding
        loss = torch.mean((channeled_z_q.detach() - channeled_z)**2) + self.beta * \
            torch.mean((channeled_z_q - channeled_z.detach()) ** 2)

        # preserve gradients
        channeled_z_q = channeled_z + (channeled_z_q - channeled_z).detach()

        # (n_batch * n_seq * n_channel) * e_dim --> n_batch * n_seq * (n_channel * e_dim)
        expand_z_q = channeled_z_q.view(n_batch, n_seq, self.n_channel * self.e_dim)

        # n_batch * (n_channel * e_dim) --> n_batch * d_model
        z_q = self.fc1(expand_z_q)

        return loss, z_q