#!/usr/bin/env python3
"""Dump NATS-Bench (NAS-Bench-201 topology space) results into lookup tables.

One JSON per (dataset, metric): ``{arch_str: value}`` over all 15,625 archs,
same key format as misc/nb201_gt.json and the nap2 lookup tables. Values are
what ``api.get_more_info(idx, dataset, iepoch=epoch-1, hp=hp, is_random=False)``
returns (seed-averaged, accuracies in percent), for every key ending in
``-accuracy`` / ``-loss`` that exists for that dataset at that epoch.

    python scripts/build_nats_lut.py --api data/NATS-tss-v1_0-3ffb9-simple \
        --epoch 20 --out data/nb201_nats_lut
"""
import argparse
import json
import os
import time

from nats_bench import create

DATASETS = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='data/NATS-tss-v1_0-3ffb9-simple')
    ap.add_argument('--epoch', type=int, default=20, help='1-based epoch (20 -> iepoch 19)')
    ap.add_argument('--hp', default='200')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    api = create(args.api, 'tss', fast_mode=True, verbose=False)
    n = len(api)
    tables = {}  # (dataset, metric) -> {arch: value}
    t0 = time.time()
    for idx in range(n):
        arch = api.arch(idx)
        for ds in DATASETS:
            info = api.get_more_info(idx, ds, iepoch=args.epoch - 1, hp=args.hp, is_random=False)
            for k, v in info.items():
                if k.endswith(('-accuracy', '-loss')) and v is not None:
                    tables.setdefault((ds, k), {})[arch] = round(float(v), 4)
        if idx % 2500 == 0:
            print(f'  {idx}/{n} ({time.time() - t0:.0f}s)', flush=True)

    names = []
    for (ds, metric), tab in sorted(tables.items()):
        name = f'ep{args.epoch}_{ds}_{metric}.json'
        with open(os.path.join(args.out, name), 'w') as f:
            json.dump(tab, f, separators=(',', ':'))
        names.append((name, len(tab)))
    readme = [f'# NATS-Bench / NAS-Bench-201 lookup tables at epoch {args.epoch} (hp={args.hp})', '',
              f'Source: `{os.path.basename(os.path.normpath(args.api))}` via `nats_bench` '
              f'(`get_more_info(idx, dataset, iepoch={args.epoch - 1}, hp="{args.hp}", is_random=False)` = mean over seeds). '
              'One JSON per (dataset, metric): `{arch_str: value}`, accuracies in percent, all 15,625 archs.', '',
              'Datasets: `cifar10-valid` = 25k train / 25k x-valid run; `cifar10` = 50k train, ori-test = the 10k test set; '
              '`cifar100`, `ImageNet16-120`: per-epoch metrics are train and valtest (x-valid + x-test); separate valid/test exist only at epoch 200.', '',
              '| file | archs |', '|---|---|'] + [f'| {nm} | {cnt} |' for nm, cnt in names]
    open(os.path.join(args.out, 'README.md'), 'w').write('\n'.join(readme) + '\n')
    print(f'wrote {len(names)} tables to {args.out} in {time.time() - t0:.0f}s')
    for nm, cnt in names:
        print(f'  {nm}: {cnt}')


if __name__ == '__main__':
    main()
