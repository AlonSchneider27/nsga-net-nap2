#!/usr/bin/env python3
"""Build misc/nb201_epoch_gt.json from the official NAS-Bench-201 .pth file.

Reads the raw torch pickle (no nas_201_api dependency):
raw['arch2infos'][idx]['full']['all_results'][(dataset, seed)] with
train_acc1es[epoch] and eval_acc1es['<split>@<epoch>'] (epoch 0-based,
accuracies in percent, hp=200 schedule, seeds 777/888/999).

For every arch and every requested epoch E (1-based; iepoch = E-1) we store the
MEAN over the seeds that have the value (== nas_201_api get_metrics with
is_random=False), for these (dataset, split) pairs:

  c10v_{train,valid,test}   cifar10-valid run (25k train / 25k x-valid; ori-test)
  c10_{train,test}          cifar10 run (50k train; ori-test = the 10k test set)
  c100_{train,valid,test,valtest}   cifar100 (x-valid, x-test, ori-test=both)
  in16_{train,valid,test,valtest}   ImageNet16-120 (same convention)

Keys: '<prefix>_<split>_ep<E>' -> float|None, plus '<prefix>_seeds' -> int.
Needs ~80 GB RAM for torch.load of the 5 GB file: run it as a CPU sbatch job.

    python scripts/build_nb201_epoch_gt.py --pth <NAS-Bench-201-v1_1-096897.pth> \
        --epochs 20,200 --check-gt misc/nb201_gt.json --out <nb201_epoch_gt.json>
"""
import argparse
import json
import os
import time

import numpy as np

DATASETS = {  # NB201 dataset key -> (prefix, {split name: eval_acc1es set name or 'train'})
    'cifar10-valid': ('c10v', {'train': 'train', 'valid': 'x-valid', 'test': 'ori-test'}),
    'cifar10': ('c10', {'train': 'train', 'test': 'ori-test'}),
    'cifar100': ('c100', {'train': 'train', 'valid': 'x-valid', 'test': 'x-test', 'valtest': 'ori-test'}),
    'ImageNet16-120': ('in16', {'train': 'train', 'valid': 'x-valid', 'test': 'x-test', 'valtest': 'ori-test'}),
}
GT_CHECK = {'c10_test_ep200': 'cifar10_test', 'c10v_valid_ep200': 'cifar10_valid',
            'c100_test_ep200': 'cifar100_test', 'in16_test_ep200': 'imagenet16_test'}


def extract(full, epochs):
    """full = arch2infos[idx]['full'] (raw state dict) -> flat dict of means over seeds."""
    out = {}
    by_ds = {}
    for (ds, seed), res in full['all_results'].items():
        by_ds.setdefault(ds, []).append(res)
    for ds, (prefix, splits) in DATASETS.items():
        results = by_ds.get(ds, [])
        out[f'{prefix}_seeds'] = len(results)
        for split, setname in splits.items():
            for E in epochs:
                ie = E - 1
                vals = []
                for res in results:
                    if setname == 'train':
                        v = res['train_acc1es'].get(ie)
                    else:
                        v = res['eval_acc1es'].get(f'{setname}@{ie}')
                    if v is not None:
                        vals.append(float(v))
                out[f'{prefix}_{split}_ep{E}'] = round(float(np.mean(vals)), 4) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pth', required=True)
    ap.add_argument('--epochs', default='20,200')
    ap.add_argument('--check-gt', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N archs')
    args = ap.parse_args()
    epochs = sorted({int(e) for e in args.epochs.split(',')})

    import torch
    t0 = time.time()
    raw = torch.load(args.pth, map_location='cpu', weights_only=False)
    print(f'loaded {args.pth} in {time.time() - t0:.0f}s; archs={len(raw["meta_archs"])}', flush=True)
    assert len(raw['meta_archs']) == 15625 and len(raw['arch2infos']) == 15625

    table = {}
    n = len(raw['meta_archs']) if not args.limit else args.limit
    for idx in range(n):
        info = raw['arch2infos'][idx]['full']
        arch = info['arch_str']
        assert arch == raw['meta_archs'][idx], (idx, arch)
        table[arch] = extract(info, epochs)
        raw['arch2infos'][idx] = None  # free memory as we go
        if idx % 2000 == 0:
            print(f'  {idx}/{n} ({time.time() - t0:.0f}s)', flush=True)
    del raw

    fields = sorted({k for v in table.values() for k in v})
    coverage = {k: sum(1 for v in table.values() if v.get(k) is not None) for k in fields}
    print('coverage (non-null per field):')
    for k in fields:
        print(f'  {k:22s} {coverage[k]}')

    validated = {'n_archs': len(table), 'n_unique': len(set(table))}
    for k in ['c10_test_ep200', 'c10v_valid_ep200', 'c100_test_ep200', 'in16_test_ep200']:
        vals = [v[k] for v in table.values() if v.get(k) is not None]
        validated[f'max_{k}'] = max(vals) if vals else None
    if args.check_gt and os.path.exists(args.check_gt):
        from scipy.stats import kendalltau
        gt = {a: v for a, v in json.load(open(args.check_gt)).items() if not a.startswith('_')}
        for ours, theirs in GT_CHECK.items():
            xs, ys = [], []
            for a, v in table.items():
                if v.get(ours) is not None and a in gt and gt[a].get(theirs) is not None:
                    xs.append(v[ours]); ys.append(gt[a][theirs])
            if len(xs) > 2:
                d = np.abs(np.array(xs) - np.array(ys))
                validated[f'vs_gt_{ours}'] = {'n': len(xs), 'kt': round(float(kendalltau(xs, ys).correlation), 4),
                                              'mean_abs_diff': round(float(d.mean()), 3), 'max_abs_diff': round(float(d.max()), 3)}
    print('validated:', json.dumps(validated, indent=1))

    meta = {'source': os.path.basename(args.pth), 'file_size': os.path.getsize(args.pth),
            'built': time.strftime('%Y-%m-%d'), 'epochs': epochs,
            'convention': "'<prefix>_<split>_ep<E>': E is 1-based (E=20 -> iepoch 19); value = mean over seeds "
                          "(777/888/999, hp=200) that report it; percent; None if no seed has it",
            'prefixes': {ds: p for ds, (p, _) in DATASETS.items()},
            'splits': {p: s for _, (p, s) in DATASETS.items()},
            'coverage': coverage, 'validated': validated}
    out = {'_meta': meta}
    out.update(table)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
