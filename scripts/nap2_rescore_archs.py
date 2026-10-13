#!/usr/bin/env python3
"""Re-score a fixed list of NB-201 architectures with one or more NAP2 predictors.

For each architecture: build the network, run ONE partial training on the
dataset (the predictors' snapshot cadence, ``steps`` snapshots), extract the
feature maps once, then encode + predict with every predictor at every prefix
length 1..steps. Mirrors the nap2 query of search/train_search.py (same model
builder, loader, seeding) but with no GA and no 20-epoch training.

Output JSON: {arch_str: {predictor_name: {"1": score, ..., "<steps>": score}},
"_meta": {...}}. The file is rewritten after every architecture, and archs
already present are skipped, so a killed chunk can simply be resubmitted.

    python scripts/nap2_rescore_archs.py --archs archs_c10.txt --dataset cifar10 \
        --predictor c10=/.../ae_lognorm_seed42_snap75/c10 \
        --predictor c100=/.../ae_lognorm_seed42_snap75/c100 \
        --predictor in16=/.../ae_lognorm_seed42_snap75/in16 \
        --steps 31 --chunk 0 --n-chunks 20 --out rescore_c10_chunk00.json
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misc.dataset_configs import build_search_transforms, get_config, get_loader_class  # noqa: E402
from nap2 import NAP2Predictor  # noqa: E402
from nap2.search_spaces.nb201_ops import build_nb201_model  # noqa: E402
from search.train_search import _LogitsFirstFromNB201, _LogitsOnly  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', required=True, help='text file, one NB-201 arch string per line')
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'ImageNet16-120'])
    ap.add_argument('--data', default='', help='dataset root override (ImageNet16-120 npy dir)')
    ap.add_argument('--predictor', action='append', required=True,
                    help='name=model_dir (dir with ae/{weights,gradients} and lstm/); repeatable')
    ap.add_argument('--steps', type=int, default=31)
    ap.add_argument('--seed', type=int, default=0, help='torch seed before each arch (train_search.main default)')
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--n-chunks', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', type=int, default=15)
    ap.add_argument('--init-channels', type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
        cudnn.enabled = True

    archs = [l.strip() for l in open(args.archs) if l.strip()]
    archs = archs[args.chunk::args.n_chunks]
    logging.info('chunk %d/%d: %d archs, dataset %s, device %s', args.chunk, args.n_chunks, len(archs), args.dataset, device)

    predictors = {}
    for spec in args.predictor:
        name, path = spec.split('=', 1)
        predictors[name] = NAP2Predictor.load(path)
    intervals = {p.snapshot_interval for p in predictors.values()}
    if len(intervals) != 1:
        raise ValueError(f'predictors disagree on snapshot_interval: {intervals}')
    interval = intervals.pop()
    logging.info('predictors %s, snapshot interval %d mb, %d steps', list(predictors), interval, args.steps)

    cfg = get_config(args.dataset)
    data_root = args.data or cfg['data_dir']
    train_data = get_loader_class(args.dataset)(root=data_root, train=True, download=True,
                                                transform=build_search_transforms(cfg, train=True))
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True,
                                              pin_memory=True, num_workers=4)

    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
        logging.info('resuming: %d archs already scored', sum(1 for k in out if not k.startswith('_')))
    out['_meta'] = {'dataset': args.dataset, 'steps': args.steps, 'snapshot_interval': interval,
                    'predictors': dict(zip(predictors, [s.split('=', 1)[1] for s in args.predictor])),
                    'seed': args.seed, 'device': device, 'chunk': args.chunk, 'n_chunks': args.n_chunks,
                    'layers': args.layers, 'init_channels': args.init_channels}
    n_cells = max(args.layers // 3, 1)
    for i, arch in enumerate(archs):
        if arch in out:
            continue
        t0 = time.time()
        model = build_nb201_model(arch, num_classes=cfg['num_classes'], C=args.init_channels, N=n_cells)
        model = _LogitsOnly(_LogitsFirstFromNB201(model))
        model.droprate = 0.0
        if device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        model = model.to(device)
        try:
            first = next(iter(predictors.values()))
            w_maps, g_maps = first.collect_maps(model, train_queue, steps=args.steps)
            t_collect = time.time() - t0
            scores = {}
            for name, pred in predictors.items():
                emb = pred.encode_maps(w_maps, g_maps)
                scores[name] = {str(k): float(pred._lstm.predict(emb[:k])) for k in range(1, args.steps + 1)}
            out[arch] = scores
            logging.info('[%d/%d] %s  collect %.0fs total %.0fs  %s', i + 1, len(archs), arch, t_collect, time.time() - t0,
                         ' '.join(f"{n}@{args.steps}={s[str(args.steps)]:.4f}" for n, s in scores.items()))
        except Exception:
            logging.exception('[%d/%d] %s FAILED', i + 1, len(archs), arch)
            out.setdefault('_failed', []).append(arch)
        finally:
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        tmp = args.out + '.tmp'
        json.dump(out, open(tmp, 'w'))
        os.replace(tmp, args.out)
    logging.info('done: %d archs in %s', sum(1 for k in out if not k.startswith('_')), args.out)


if __name__ == '__main__':
    main()
