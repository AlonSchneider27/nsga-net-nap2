#!/usr/bin/env python3
"""Zero-cost proxy lookup tables (synflow, grad_norm, snip) for a list of NB-201 archs on one dataset.

Mirrors the live guided path (search/train_search.py): the untrained network (same builder, same seeding) is
scored at initialization from ONE minibatch of the search train loader by fitness.zero_cost. Budget-free.
Output JSON {arch_str: {synflow: v, grad_norm: v, snip: v}}, rewritten after every arch; resume-safe.

    python scripts/zc_rescore_archs.py --archs archs_all.txt --dataset cifar10 --chunk 0 --n-chunks 20 --out zc_cifar10_chunk00.json
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

import fitness as fitness_pkg  # noqa: E402
from misc.dataset_configs import build_search_transforms, get_config, get_loader_class  # noqa: E402
from nap2.search_spaces.nb201_ops import build_nb201_model  # noqa: E402
from search.train_search import _LogitsFirstFromNB201, _LogitsOnly  # noqa: E402

METHODS = ['synflow', 'grad_norm', 'snip']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', required=True)
    ap.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'ImageNet16-120'])
    ap.add_argument('--data', default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--n-chunks', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', type=int, default=15)
    ap.add_argument('--init-channels', type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True; cudnn.enabled = True
    archs = [l.strip() for l in open(args.archs) if l.strip()][args.chunk::args.n_chunks]
    cfg = get_config(args.dataset)
    train_data = get_loader_class(args.dataset)(root=args.data or cfg['data_dir'], train=True, download=True,
                                                transform=build_search_transforms(cfg, train=True))
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
    scorers = fitness_pkg.build_scorers(','.join(METHODS))
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    out['_meta'] = {'dataset': args.dataset, 'seed': args.seed, 'methods': METHODS, 'chunk': args.chunk, 'n_chunks': args.n_chunks,
                    'layers': args.layers, 'init_channels': args.init_channels, 'device': device}
    n_cells = max(args.layers // 3, 1)
    logging.info('chunk %d/%d: %d archs, %s, %s', args.chunk, args.n_chunks, len(archs), args.dataset, device)
    for i, arch in enumerate(archs):
        if arch in out:
            continue
        t0 = time.time()
        if device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        model = _LogitsOnly(_LogitsFirstFromNB201(build_nb201_model(arch, num_classes=cfg['num_classes'], C=args.init_channels, N=n_cells)))
        model.droprate = 0.0
        model = model.to(device)
        try:
            inputs, targets = next(iter(train_queue))
            inputs, targets = inputs.to(device), targets.to(device)
            scores = {}
            for s in scorers:
                try:
                    scores[s.name] = float(s.score_init(model, inputs, targets))
                except Exception:
                    logging.exception('%s failed on %s', s.name, arch); scores[s.name] = None
            out[arch] = scores
            logging.info('[%d/%d] %s %.1fs %s', i + 1, len(archs), arch, time.time() - t0,
                         ' '.join(f'{k}={v:.4g}' if v is not None else f'{k}=None' for k, v in scores.items()))
        except Exception:
            logging.exception('[%d/%d] %s FAILED', i + 1, len(archs), arch)
            out.setdefault('_failed', []).append(arch)
        finally:
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        tmp = args.out + '.tmp'; json.dump(out, open(tmp, 'w')); os.replace(tmp, args.out)
    logging.info('done: %d archs', sum(1 for k in out if not k.startswith('_')))


if __name__ == '__main__':
    main()
