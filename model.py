import torch
import torch.nn as nn
import torch.nn.functional as F

from module.decoder import Decoder
from module.encoder import Encoder
from module.f0 import compute_f0
from module.match_features import match_features


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()
