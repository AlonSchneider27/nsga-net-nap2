#!/usr/bin/env python3
"""Dump NATS-Bench per-epoch learning curves (epochs 1..E, hp=200, seed-averaged) for all 15,625 NB-201 archs.

Output: <out>/<ds>.json = {arch_str: {"train_loss": [E], "val_acc": [E]}} (val_acc in percent) with
  c10  <- NATS 'cifar10-valid' run: train-loss + valid-accuracy (x-valid, the true validation split)
  c100 <- NATS 'cifar100':        train-loss + valtest-accuracy (per-epoch NATS only has x-valid+x-test combined)
  in16 <- NATS 'ImageNet16-120':  train-loss + valtest-accuracy
These feed scripts/build_lc_tables.py (SoTL, SoTL-E, Early-Stop, LCE-m, LC-PFN tables at every epoch budget).

    .venv/bin/python scripts/build_nats_curves.py --api data/NATS-tss-v1_0-3ffb9-simple --epochs 20 --out data/nb201_nats_curves
"""
import argparse
import json
import os
import time

from nats_bench import create

SRC = {'c10': ('cifar10-valid', 'valid-accuracy'), 'c100': ('cifar100', 'valtest-accuracy'),
       'in16': ('ImageNet16-120', 'valtest-accuracy')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='data/NATS-tss-v1_0-3ffb9-simple')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    api = create(args.api, 'tss', fast_mode=True, verbose=False)
    n = len(api); t0 = time.time()
    curves = {ds: {} for ds in SRC}
    for idx in range(n):
        arch = api.arch(idx)
        for ds, (nats_ds, acc_key) in SRC.items():
            tl, va = [], []
            for e in range(args.epochs):
                info = api.get_more_info(idx, nats_ds, iepoch=e, hp='200', is_random=False)
                tl.append(round(float(info['train-loss']), 5)); va.append(round(float(info[acc_key]), 4))
            curves[ds][arch] = {'train_loss': tl, 'val_acc': va}
        if idx % 1000 == 0:
            print(f'  {idx}/{n} ({time.time() - t0:.0f}s)', flush=True)
    for ds in SRC:
        with open(os.path.join(args.out, f'{ds}.json'), 'w') as f:
            json.dump(curves[ds], f, separators=(',', ':'))
    meta = {'source': os.path.basename(os.path.normpath(args.api)), 'hp': '200', 'is_random': False,
            'epochs': args.epochs, 'datasets': {ds: {'nats_dataset': s[0], 'val_key': s[1]} for ds, s in SRC.items()},
            'note': 'train_loss = NATS per-epoch mean minibatch train loss (seed average); val_acc in percent'}
    json.dump(meta, open(os.path.join(args.out, 'meta.json'), 'w'), indent=1)
    print(f'wrote {len(SRC)} curve files ({n} archs x {args.epochs} epochs) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
