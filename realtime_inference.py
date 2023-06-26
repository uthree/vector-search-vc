import argparse
import os

import torch
import torchaudio
from torchaudio.functional import resample as resample

import numpy as np
import pyaudio

from model import VoiceConvertor, match_features
from module.hubert import load_hubert, extract_hubert_feature
from module.f0 import compute_f0

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-t', '--target', default='./target.wav',
                    help="Target voice")
parser.add_argument('-c', '--chunk', default=3072, type=int)
parser.add_argument('-b', '--buffer', default=6, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-ic', '--inputchannels', default=1, type=int)
parser.add_argument('-oc', '--outputchannels', default=1, type=int)
parser.add_argument('-lc', '--loopbackchannels', default=1, type=int)
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-f0', '--f0-rate', default=1.0, type=float)
parser.add_argument('-amp', default=1.0, type=float)

args = parser.parse_args()

device = torch.device(args.device)

vc = VoiceConvertor().to(device)
vc.load_state_dict(torch.load('./model.pt', map_location=device))

hubert = load_hubert(device)

print("Loading target...")
wf, sr = torchaudio.load(args.target, normalize=True)
wf = wf.to(device)
wf = resample(wf, sr, 16000)
# encode speaker
target_feature = extract_hubert_feature(hubert, wf)

audio = pyaudio.PyAudio()
stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=44100,
        channels=args.inputchannels,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.outputchannels,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=44100, 
        channels=args.loopbackchannels,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

print("Converting Voice...")


buffer = []
chunk = args.chunk
buffer_size = args.buffer
while True:
    data = stream_input.read(chunk, exception_on_overflow=False)
    data = np.frombuffer(data, dtype=np.int16)
    buffer.append(data)
    if len(buffer) > buffer_size:
        del buffer[0]
    else:
        continue
    data = np.concatenate(buffer, 0)
    data = data.astype(np.float32) / 32768
    data = torch.from_numpy(data).to(device)
    data = torch.unsqueeze(data, 0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wf = resample(data, 44100, 16000)

            # Convert
            f0 = compute_f0(wf) * args.f0_rate
            hubert_feature = extract_hubert_feature(hubert, wf)
            hubert_feature = match_features(hubert_feature, target_feature)
            z = vc.encoder(hubert_feature, f0)
            wf = vc.decoder(z)

            data = resample(wf, 16000, 44100)
            data = data.squeeze(0)

    data = data.cpu().numpy()
    data = data * 32768 * args.amp
    data = data.astype(np.int16)
    s = (chunk * buffer_size) // 2 - (chunk // 2)
    e = (chunk * buffer_size) - s
    data = data[s:e]
    data = data.tobytes()
    stream_output.write(data)
    if stream_loopback is not None:
        stream_loopback.write(data)
