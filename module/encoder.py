import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, wavlm_dim=1024, hifigan_dim=512):
        super().__init__()
        self.proj = nn.Conv1d(wavlm_dim, hifigan_dim, 1, 1, 0, bias=False)
        self.f0_enc = F0Encoder(hifigan_dim)

    def forward(self, wavlm_feature, f0):
        return self.proj(wavlm_feature) + self.f0_enc(f0)


class F0Encoder(nn.Module):
    def __init__(self, hifigan_dim=512):
        super().__init__()
        self.c1 = nn.Conv1d(1, hifigan_dim, 1, 1, 0)
        self.c2 = nn.Conv1d(hifigan_dim, hifigan_dim, 1, 1, 0)
        self.c1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.c1(x)
        x = torch.sin(x)
        x = self.c2(x)
        return x
