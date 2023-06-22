import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

from model import VoiceConvertorWrapper

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu',
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-i', '--input', default='./inputs',
                    help="Input directory")
parser.add_argument('-o', '--output', default='./outputs',
                    help="Output directory")
parser.add_argument('-t', '--target', default='./target.wav')
parser.add_argument('-f0', '--f0-rate', default=1.0, type=float)


args = parser.parse_args()

device = torch.device(args.device)

convertor = VoiceConvertorWrapper('model.pt', device=device)

if not os.path.exists(args.output):
    os.mkdir(args.output)

print("Encoding target speaker...")
wf, sr = torchaudio.load(args.target)
wf = resample(wf, sr, 16000)
wf = wf.to(device)
spk = convertor.get_speaker_index(wf)


for i, fname in enumerate(os.listdir(args.input)):
    print(f"Inferencing {fname}")
    with torch.no_grad():
        wf, sr = torchaudio.load(os.path.join(args.input, fname))
        wf = resample(wf, sr, 16000)
        wf = wf.to(device)
        
        wf = convertor.convert(wf, spk, f0_rate=args.f0_rate, k=4)
        
        wf = resample(wf, 16000, sr)
        wf = wf.to(torch.device('cpu'))
        out_path = os.path.join(args.output, f"output_{fname}_{i}.wav")
        torchaudio.save(out_path, src=wf, sample_rate=sr)



