import torch
import torch.nn as nn
import numpy as np
import faiss

from module.decoder import Decoder
from module.discriminator import Discriminator
from module.encoder import Encoder
from module.hubert import load_hubert, interpolate_hubert_output
from module.f0 import compute_f0


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()


class Speaker():
    def __init__(self, index, dictionary):
        self.index = index
        self.dictionary = dictionary


class VoiceConvertorWrapper():
    def __init__(self, model_path='model.pt', device=torch.device('cpu')):
        super().__init__()
        self.convertor = VoiceConvertor().to(device)
        self.convertor.load_state_dict(torch.load(model_path, map_location=device))
        self.device = device
        self.hubert = load_hubert(device)

    def get_speaker_index(self, wave):
        index = faiss.IndexFlatIP(768)
        with torch.no_grad():
            if wave.ndim == 2:
                wave = wave[0]
            wave = wave.unsqueeze(0)
            features = interpolate_hubert_output(self.hubert(wave), wave.shape[1])
            features = features.squeeze(0)
            features = features.transpose(0, 1)
            features = features.cpu().numpy()
            index.add(features)
        return Speaker(index, features)

    def convert_features(self, features, speaker, k=4, alpha=0.1):
        features_in = features
        features = features[0]
        # [768, length]
        features = features.transpose(0, 1) # [length, 768]
        features = features.cpu().numpy()
        D, I = speaker.index.search(features, k)
        features = np.mean(speaker.dictionary[I], axis=1)
        features = torch.from_numpy(features).to(self.device)
        features = features.transpose(0, 1)
        features = features.unsqueeze(0)
        features_in * alpha + features * (1 - alpha)
        return features

    def convert(self, wave, speaker, f0_rate=1.0, k=4, alpha=0.1):
        index = speaker.index
        with torch.no_grad():
            if wave.ndim == 2:
                wave = wave[0]
            wave = wave.unsqueeze(0) # [1, Length]
            hubert_features = interpolate_hubert_output(self.hubert(wave), wave.shape[1]) # [1, 768, Length]
            f0 = compute_f0(wave[0]).unsqueeze(0) * f0_rate
            hubert_features = self.convert_features(hubert_features, speaker, k, alpha)
            z = self.convertor.encoder.encode(hubert_features, f0)
            return self.convertor.decoder(z)
