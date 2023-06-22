import torch
import torch.nn.functional as F
from transformers import HubertModel

def load_hubert(device=torch.device('cpu')):
    print("Loading HuBERT...")
    model = HubertModel.from_pretrained("rinna/japanese-hubert-base").to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def interpolate_hubert_output(hubert_output, wave_length):
    c = hubert_output.last_hidden_state
    c = c.transpose(1, 2)
    c = F.interpolate(c, size=(wave_length // 256))
    return c
