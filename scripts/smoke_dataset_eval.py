#!/usr/bin/env python3
"""Smoke test of nap2 vs the fitness baselines on a dataset without lookup tables.

For a fixed list of NB-201 archs, runs the project's native per-arch pipeline (search/train_search.main) once per
arch: zero-cost proxies at init, nap2 (one predictor, --nap2-steps snapshots), the LC methods at their native epoch
cadence from the real training loop, and the validation accuracy after --epochs epochs as the reference. Chunked
and resume-safe. Output JSON {arch: {valid_acc, pred_acc, flops, params, fitness_scores}}.

    python scripts/smoke_dataset_eval.py --archs archs.txt --dataset spherical_cifar100 --data <dir> --epochs 8 \
        --predictor <ae_lognorm_seed42_snap75/c100> --lcpfn-ckpt <pt> --chunk 0 --n-chunks 7 --out <json> --expr-root <dir>
"""
import argparse
import json
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitness as fitness_pkg  # noqa: E402
from models.micro_genotypes import NB201_PRIMITIVES  # noqa: E402
from nap2 import NAP2Predictor  # noqa: E402
from search import train_search  # noqa: E402

OP_RE = re.compile(r'([a-z_0-9]+)~\d')


def genome_of(arch_str):
    ops = OP_RE.findall(arch_str)
    assert len(ops) == 6, arch_str
    return [NB201_PRIMITIVES.index(o) for o in ops]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data', default='')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--predictor', default='', help='nap2 model dir (ae/{weights,gradients}, lstm/); empty = no nap2')
    ap.add_argument('--nap2-steps', type=int, default=5)
    ap.add_argument('--fitness', default='all')
    ap.add_argument('--lcpfn-ckpt', default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--n-chunks', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expr-root', required=True)
    args = ap.parse_args()
    archs = [l.strip() for l in open(args.archs) if l.strip()][args.chunk::args.n_chunks]
    pred = NAP2Predictor.load(args.predictor) if args.predictor else None
    scorers = fitness_pkg.build_scorers(args.fitness, lcpfn_ckpt=args.lcpfn_ckpt, target_epochs=args.epochs)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    out['_meta'] = {'dataset': args.dataset, 'epochs': args.epochs, 'predictor': args.predictor, 'nap2_steps': args.nap2_steps,
                    'fitness': [s.name for s in scorers], 'seed': args.seed, 'chunk': args.chunk, 'n_chunks': args.n_chunks}
    for i, arch in enumerate(archs):
        if arch in out:
            continue
        t0 = time.time()
        perf = train_search.main(genome_of(arch), epochs=args.epochs, search_space='nb201',
                                 save=f'chunk{args.chunk}_arch{i}', expr_root=args.expr_root, seed=args.seed,
                                 init_channels=16, layers=15, predictor=pred, dataset=args.dataset, data=args.data,
                                 nap2_steps=args.nap2_steps, fitness_scorers=scorers, lc_cadence='epoch')
        out[arch] = {'valid_acc': perf['valid_acc'], 'pred_acc': perf['pred_acc'], 'flops': perf['flops'],
                     'params': perf['params'], 'fitness_scores': perf['fitness_scores'], 'time_s': round(time.time() - t0, 1)}
        tmp = args.out + '.tmp'; json.dump(out, open(tmp, 'w'), indent=0); os.replace(tmp, args.out)
        logging.info('SMOKE [%d/%d] %s valid_acc=%.2f nap2=%s %.0fs', i + 1, len(archs), arch, perf['valid_acc'],
                     perf['pred_acc'], time.time() - t0)
    logging.info('done: %d archs', sum(1 for k in out if not k.startswith('_')))


if __name__ == '__main__':
    main()
