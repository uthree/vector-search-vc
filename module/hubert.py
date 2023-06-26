import torch
import torch.nn.functional as F
from transformers import HubertModel

def load_hubert(device=torch.device('cpu')):
    print("Loading HuBERT...")
    model = HubertModel.from_pretrained("rinna/japanese-hubert-base").to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_hubert_feature(hubert, wave):
    length = wave.shape[1] // 320
    feature = hubert(wave, output_hidden_states=True).hidden_states[6]
    feature = feature.transpose(1, 2)
    feature = F.interpolate(feature, length, mode='linear')
    return feature
