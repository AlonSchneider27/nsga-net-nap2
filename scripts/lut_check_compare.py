#!/usr/bin/env python3
"""Sanity check: table-backed guided nap2 run (--nap2_lookup) vs the live guided nap2 run, same seed/budget.

    .venv/bin/python scripts/lut_check_compare.py --live <seed dir with in_dataset/<ds>/*nap2x75*> \
        --lut <dir with *nap2lut31*> --gt data/nb201_nats_gt.json --lut-tables <nb201_lut_snap75_31/tables>

Prints, per dataset, the per-seed summary metrics of both runs (pick@k, oracle@20, fidelity KT, unique ratio),
the overlap of the visited sets, and on the common archs the agreement between live and table nap2 scores.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guided_summary_html import analyse, KEY  # noqa: E402

DS = {'c10': 'cifar10', 'c100': 'cifar100', 'in16': 'imagenet16'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', required=True)
    ap.add_argument('--lut', required=True)
    ap.add_argument('--gt', default='data/nb201_nats_gt.json')
    ap.add_argument('--lut-tables', default='', help='tables/ dir: adds table score of every live-visited arch')
    args = ap.parse_args()
    gt_all = {a: v for a, v in json.load(open(args.gt)).items() if not a.startswith('_')}
    cols = ['n_unique', 'unique_ratio', 'pick@1', 'pick@5', 'pick@10', 'pick@20', 'oracle@20', 'best_visited', 'fid_kt', 'fid_kt_top10']
    for ds, tname in DS.items():
        gt = {a: v[KEY[ds]] for a, v in gt_all.items()}
        live = glob.glob(os.path.join(args.live, 'in_dataset', ds, '*nap2x75*'))
        lut = glob.glob(os.path.join(args.lut, f'*{ds}_nap2lut31*'))
        if not live or not lut:
            print(ds, 'missing run', live, lut); continue
        L, T = analyse(live[0], ds, ds, gt), analyse(lut[0], ds, ds, gt)
        print(f'\n== {ds}')
        print(f"{'':14}" + ''.join(f'{c:>13}' for c in cols))
        for name, r in (('live', L), ('table', T)):
            print(f'{name:14}' + ''.join(f'{r[c]:>13}' for c in cols))
        sl, st = set(L['_archs']), set(T['_archs'])
        common = sl & st
        print(f'visited overlap: {len(common)} common of {len(sl)} live / {len(st)} table (Jaccard {len(common)/len(sl|st):.2f})')
        top20 = set(np.array(L['_archs'])[np.argsort(-L['_scores'])[:20]]) & set(np.array(T['_archs'])[np.argsort(-T['_scores'])[:20]])
        print(f'top-20 picks shared: {len(top20)}')
        if args.lut_tables:
            tab = json.load(open(os.path.join(args.lut_tables, f'trained_on_{tname}', f'nap2_snap5_{tname}.json')))
            ls = np.array(L['_scores']); ts = np.array([tab[a] for a in L['_archs']])
            print(f'live vs table nap2 score on the {len(ls)} live-visited archs: KT {kendalltau(ls, ts)[0]:.3f}, '
                  f'mean |diff| {np.abs(ls - ts).mean():.4f}, live std {ls.std():.4f}; '
                  f'KT vs GT live {kendalltau(ls, L["_gt"])[0]:.3f} / table {kendalltau(ts, L["_gt"])[0]:.3f}')


if __name__ == '__main__':
    main()
