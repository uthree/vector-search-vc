import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hubert_dim=768, embedding_dim=192):
        super().__init__()
        self.feat_enc = nn.Conv1d(hubert_dim, embedding_dim, 1)
        self.f0_enc = nn.Conv1d(1, embedding_dim, 1)
        self.decode_conv = nn.Sequential(
                nn.Conv1d(embedding_dim, embedding_dim * 4, 1),
                nn.ReLU(),
                nn.Conv1d(embedding_dim * 4, embedding_dim * 2, 1))
        

    def forward(self, hubert_feature, f0):
        x = self.feat_enc(hubert_feature) + torch.sin(self.f0_enc(f0))
        mu, sigma = torch.chunk(self.decode_conv(x), 2, dim=1)
        return mu, sigma

    def encode(self, hubert_feature, f0):
        mu, sigma = self.forward(hubert_feature, f0)
        return mu

