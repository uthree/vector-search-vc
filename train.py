import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from model import VoiceConvertor
from module.match_features import match_features
from module.dataset import WaveFileDirectory
from module.discriminator import Discriminator, MelSpectrogramLoss
from module.wavlm import load_wavlm, extract_wavlm_feature
from module.f0 import compute_f0

parser = argparse.ArgumentParser(description="run training")

parser.add_argument('dataset')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float)
parser.add_argument('-len', '--length', default=7040, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-gacc', '--gradient-accumulation', default=1, type=int)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    vc = VoiceConvertor().to(device)
    disc = Discriminator().to(device)
    if os.path.exists('./model.pt'):
        vc.load_state_dict(torch.load('./model.pt', map_location=device))
    if os.path.exists('./discriminator.pt'):
        disc.load_state_dict(torch.load('./discriminator.pt', map_location=device))
    return vc, disc


def save_models(vc, disc):
    print("Saving Models...")
    torch.save(vc.state_dict(), './model.pt')
    torch.save(disc.state_dict(), './discriminator.pt')
    print("complete!")


def write_preview(source_wave, file_path='./preview.wav'):
    source_wave = source_wave.detach().to(torch.float32).cpu()
    torchaudio.save(src=source_wave, sample_rate=16000, filepath=file_path)

device = torch.device(args.device)
vc, D = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptC = optim.AdamW(vc.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=0.01)

scheduler = optim.lr_scheduler.StepLR(OptC, 1, gamma=0.999)

wavlm = load_wavlm(device)

grad_acc = args.gradient_accumulation

mel_loss = MelSpectrogramLoss().to(device)

weight_fm = 2.0
weight_mel = 45.0

OptC.zero_grad()

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device)
        N = wave.shape[0]
        
        # Train Convertor.
        with torch.cuda.amp.autocast(enabled=args.fp16):
            f0 = compute_f0(wave)

            feature = extract_wavlm_feature(wavlm, wave)

            z = vc.encoder(feature, f0)
            wave_out = vc.decoder(z)
            loss_adv = 0
            logits = D.logits(wave_out)
            for logit in logits:
                loss_adv += (logit ** 2).mean() / len(logits)

            loss_fm = D.feat_loss(wave_out, wave)
            loss_mel = mel_loss(wave_out, wave)

            loss_c = loss_adv + weight_fm * loss_fm + weight_mel * loss_mel

        scaler.scale(loss_c).backward()

        if batch % grad_acc == 0:
            scaler.step(OptC)
            OptC.zero_grad()

        # Train Discriminator.
        OptD.zero_grad()
        wave_out = wave_out.detach() # Fake wave
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_d = 0
            logits = D.logits(wave_out)
            for logit in logits:
                loss_d += ((logit - 1) ** 2).mean() / len(logits)
            logits = D.logits(wave)
            for logit in logits:
                loss_d += (logit ** 2).mean() / len(logits)
        loss_d = loss_d / 2
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        
        tqdm.write(f"D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, F.M.: {loss_fm.item():.4f}, Mel.: {loss_mel.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 200 == 0:
            save_models(vc, D)

    scheduler.step()


print("Training Complete!")
save_models(vc, D)
