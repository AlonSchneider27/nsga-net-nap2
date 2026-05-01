"""Standalone nap2 smoke test: load predictor + score one NetworkCIFAR architecture.

Bypasses the evolutionary search entirely. Useful for verifying nap2 + the
trained_models/ checkpoints work on this machine before debugging the
integration in train_search.py.

Run from project root:
    .venv/bin/python scripts/smoke_nap2.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from nap2 import NAP2Predictor
from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.lstm_predictor import LSTMPredictor

import search.cifar10_search as my_cifar10
from models.micro_models import NetworkCIFAR
from models import micro_genotypes


CHECKPOINT_DIR = 'trained_models/cifar10'
DATA_ROOT = 'data'
BATCH_SIZE = 128
INIT_CHANNELS = 16
LAYERS = 8
NUM_CLASSES = 10
NAP2_STEPS = 5


class _LogitsOnly(nn.Module):
    """NetworkCIFAR returns (logits, aux); nap2 takes outputs[-1]. Adapt."""

    def __init__(self, m):
        super().__init__()
        self.inner = m

    def forward(self, x):
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out


def load_predictor() -> NAP2Predictor:
    base = CHECKPOINT_DIR
    ae_w = FeatureMapAutoEncoder.load(
        model_path=f'{base}/ae/weights/ae_weights.pt',
        params_path=f'{base}/ae/weights/aew_model_hyper_params.json',
    )
    ae_g = FeatureMapAutoEncoder.load(
        model_path=f'{base}/ae/gradients/ae_gradients.pt',
        params_path=f'{base}/ae/gradients/aeg_model_hyper_params.json',
    )
    lstm = LSTMPredictor.load(
        model_path=f'{base}/lstm/cp/model_state_cp/lstm_reg_final.pt',
        params_path=f'{base}/lstm/lstm_model_hyper_params.json',
    )
    return NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=lstm, normalize='none')


def build_dataloader() -> torch.utils.data.DataLoader:
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    train_data = my_cifar10.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=False, num_workers=0,
    )


def build_model() -> nn.Module:
    genotype = micro_genotypes.NSGANet
    model = NetworkCIFAR(INIT_CHANNELS, NUM_CLASSES, LAYERS, auxiliary=False, genotype=genotype)
    model.droprate = 0.0
    return model


def main() -> int:
    print('[1/4] loading predictor...', flush=True)
    t0 = time.time()
    predictor = load_predictor()
    print(f'      done in {time.time() - t0:.1f}s', flush=True)

    print('[2/4] building dataloader...', flush=True)
    t0 = time.time()
    loader = build_dataloader()
    print(f'      done in {time.time() - t0:.1f}s ({len(loader.dataset)} samples)', flush=True)

    print('[3/4] building NetworkCIFAR (init_channels=16, layers=8)...', flush=True)
    t0 = time.time()
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'      done in {time.time() - t0:.1f}s ({n_params:.3f}M params)', flush=True)

    model = _LogitsOnly(model)

    print(f'[4/4] scoring with predictor.score(steps={NAP2_STEPS})...', flush=True)
    t0 = time.time()
    try:
        score = float(predictor.score(model, loader, steps=NAP2_STEPS))
    except Exception:
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0
    print(f'      done in {elapsed:.1f}s')
    print(f'\nnap2 score = {score:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
