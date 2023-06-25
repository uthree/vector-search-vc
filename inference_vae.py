import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

from model import VoiceConvertor
from module.f0 import compute_f0
from module.hubert import load_hubert, extract_hubert_feature

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-f0', '--f0-rate', default=1.0, type=float)
args = parser.parse_args()

device = torch.device(args.device)

vc = VoiceConvertor().to(device)
vc.load_state_dict(torch.load('./model.pt', map_location=device))

hubert = load_hubert(device)

if not os.path.exists(args.output):
    os.mkdir(args.output)


for i, fname in enumerate(os.listdir(args.input)):
    print(f"Inferencing {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 16000)
        wf = wf.to(device)
        
        f0 = compute_f0(wf) * args.f0_rate
        hubert_feature = extract_hubert_feature(hubert, wf)
        z = vc.encoder(hubert_feature, f0)
        wf = vc.decoder(z)

        wf = resample(wf, 16000, sr)
        wf = wf.to(torch.device('cpu'))
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)

