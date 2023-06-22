import torch
import torch.nn as nn

from module.decoder import Decoder
from module.discriminator import Discriminator
from module.encoder import Encoder


class VoiceConvertor(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()


class VoiceConvertorWrapper():
    def __init__(self, model_path='model.pt', device=torch.device('cpu')):
        super().__init__()
        # load model
