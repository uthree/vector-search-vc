import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d, AvgPool1d, Conv2d
import torchaudio


LRELU_SLOPE = 0.1


class PeriodicDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.layers = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1)),
        ])
        self.output_layer = norm_f(Conv2d(1024, 1, (3, 1), 1))

    def forward(self, x):
        # padding
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        for layer in self.layers:
            x = layer(x)
            F.leaky_relu(x, LRELU_SLOPE)
        x = self.output_layer(x)
        return x

    def feat(self, x):
        # padding
        fmap = []
        if x.shape[1] % self.period != 0:
            pad_len = self.period - (x.shape[1] % self.period)
            x = torch.cat([x, torch.zeros(x.shape[0], pad_len, device=x.device)], dim=1)

        x = x.view(x.shape[0], self.period, -1)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        for layer in self.layers:
            x = layer(x)
            F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        return fmap


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([])
        for p in periods:
            self.sub_discriminators.append(PeriodicDiscriminator(p))

    def forward(self, x):
        logits = []
        for sd in self.sub_discriminators:
            logits.append(sd(x))
        return logits

    def feat(self, x):
        feats = []
        for sd in self.sub_discriminators:
            feats += sd(x)
        return feats


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.layers = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.post(x)
        return x

    def feat(self, x):
        x = x.unsqueeze(1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feats.append(x)
        x = self.post(x)
        return feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        logits = []
        for sd, pool in zip(self.sub_discriminators, self.pools):
            logits.append(sd(pool(x)))
        return logits

    def feat(self, x):
        feats = []
        for sd, pool in zip(self.sub_discriminators, self.pools):
            feats += sd.feat(pool(x))
        return feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator()
        self.MSD = MultiScaleDiscriminator()

    def logits(self, x):
        return self.MPD(x) + self.MSD(x)

    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.MPD.feat(real) + self.MSD.feat(real)
        fake_feat = self.MPD.feat(fake) + self.MSD.feat(fake)
        loss = 0
        for r, f in zip(real_feat, fake_feat):
            loss = loss + F.l1_loss(f, r)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=16000, n_ffts=[1024], n_mels=80, normalized=False):
        super().__init__()
        self.to_mels = nn.ModuleList([])
        for n_fft in n_ffts:
            self.to_mels.append(torchaudio.transforms.MelSpectrogram(sample_rate,
                                                                n_mels=n_mels,
                                                                n_fft=n_fft,
                                                                normalized=normalized,
                                                                hop_length=256))

    def forward(self, fake, real):
        loss = 0
        for to_mel in self.to_mels:
            to_mel = to_mel.to(real.device)
            with torch.no_grad():
                real_mel = to_mel(real)
            loss += F.l1_loss(to_mel(fake), real_mel).mean() / len(self.to_mels)
        return loss
