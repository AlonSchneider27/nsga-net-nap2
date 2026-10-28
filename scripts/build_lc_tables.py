#!/usr/bin/env python3
"""Learning-curve baseline lookup tables from NATS-Bench per-epoch curves (epoch budgets 1..E).

Reads <curves>/<ds>.json from scripts/build_nats_curves.py and scores every arch with the repo's own
scorers at their NATIVE epoch cadence (fitness.trace.epoch_native_scores: sotl = -(sum of epoch losses),
sotl_e = -(last epoch loss), early_stop = val acc at epoch e, lce_m / lc_pfn = extrapolation of the first e
val-acc points to --target-epochs). Nothing about the methods is changed; one "minibatch" is one epoch.

Chunked (lce_m runs a 300-step MCMC per prefix, ~4.5 s per arch for 20 prefixes) and resume-safe:
    python scripts/build_lc_tables.py --curves data/nb201_nats_curves --dataset c10 --chunk 0 --n-chunks 100 \
        --lcpfn-ckpt trained_models/lcpfn/pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt --out <dir>/chunks
    python scripts/build_lc_tables.py --finalize --curves ... --out <dir>      # merge chunks -> tables/<method>/<method>_e<E>_<ds>.json
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitness as fitness_pkg  # noqa: E402
from fitness.trace import epoch_native_scores  # noqa: E402

METHODS = ['sotl', 'sotl_e', 'early_stop', 'lce_m', 'lc_pfn']
DS = ['c10', 'c100', 'in16']


def finalize(args):
    archs_order = None
    for ds in DS:
        merged = {}
        for f in sorted(glob.glob(os.path.join(args.out, 'chunks', f'lc_{ds}_chunk*.json'))):
            merged.update({k: v for k, v in json.load(open(f)).items() if not k.startswith('_')})
        curves = json.load(open(os.path.join(args.curves, f'{ds}.json')))
        missing = [a for a in curves if a not in merged]
        if missing:
            print(f'{ds}: {len(missing)} archs missing, not finalized'); continue
        n_ep = len(next(iter(curves.values()))['val_acc'])
        for m in METHODS:
            d = os.path.join(args.out, 'tables', m); os.makedirs(d, exist_ok=True)
            for e in range(1, n_ep + 1):
                tab = {a: merged[a][m][str(e)] for a in curves}
                with open(os.path.join(d, f'{m}_e{e}_{ds}.json'), 'w') as fh:
                    json.dump(tab, fh, separators=(',', ':'))
        print(f'{ds}: {len(merged)} archs -> {len(METHODS) * n_ep} tables')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', required=True)
    ap.add_argument('--dataset', choices=DS)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--n-chunks', type=int, default=1)
    ap.add_argument('--target-epochs', type=int, default=200, help='extrapolation horizon for lce_m / lc_pfn (NATS GT = epoch 200)')
    ap.add_argument('--lcpfn-ckpt', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)  # the scorers' per-call fallback warnings are expected here
    if args.finalize:
        return finalize(args)
    os.makedirs(os.path.join(args.out, 'chunks'), exist_ok=True)
    out_path = os.path.join(args.out, 'chunks', f'lc_{args.dataset}_chunk{args.chunk:03d}.json')
    curves = json.load(open(os.path.join(args.curves, f'{args.dataset}.json')))
    archs = list(curves)[args.chunk::args.n_chunks]
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out['_meta'] = {'dataset': args.dataset, 'target_epochs': args.target_epochs, 'methods': METHODS,
                    'chunk': args.chunk, 'n_chunks': args.n_chunks, 'source': 'NATS per-epoch curves (hp=200, seed avg)'}
    scorers = fitness_pkg.build_scorers(','.join(METHODS), lcpfn_ckpt=args.lcpfn_ckpt, target_epochs=args.target_epochs)
    t0 = time.time(); done = 0
    for i, a in enumerate(archs):
        if a in out:
            continue
        c = curves[a]
        s = epoch_native_scores(scorers, c['train_loss'], [v / 100.0 for v in c['val_acc']])
        out[a] = {m: {k.split('@e')[1]: v for k, v in s.items() if k.startswith(m + '@e')} for m in METHODS}
        done += 1
        if done % 10 == 0 or i == len(archs) - 1:
            tmp = out_path + '.tmp'; json.dump(out, open(tmp, 'w')); os.replace(tmp, out_path)
            print(f'[{i + 1}/{len(archs)}] {time.time() - t0:.0f}s', flush=True)
    tmp = out_path + '.tmp'; json.dump(out, open(tmp, 'w')); os.replace(tmp, out_path)
    print(f'done chunk {args.chunk}: {sum(1 for k in out if not k.startswith("_"))} archs')


if __name__ == '__main__':
    main()
