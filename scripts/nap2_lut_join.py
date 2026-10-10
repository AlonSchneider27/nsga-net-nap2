#!/usr/bin/env python3
"""Join the 75-mb NAP2 lookup tables onto finished native runs (no GPU).

For every native run (in-dataset or cross-dataset) the unique architectures are
looked up in the table of the run's predictor dataset x target dataset at every
step k (75*k mini-batches). Writes, per run, ``nap2_snap75_lut.json``
({arch_str: {"k": score}} + _meta) and a comparison of Kendall tau against the
NB-201 200-epoch ground truth and the run's own 20-epoch valid_acc, for the
75-mb lookup scores and the 100-mb live nap2 scores already in summary.json.

The 750 architectures the 75-mb predictors were fitted on (arch_index.json
split train/val) are flagged; KT is reported on all visited archs and on the
clean subset. Nothing is removed from the runs.

    python scripts/nap2_lut_join.py --runs data/research/native/seed_42 \
        --lut data/nb201_lut_snap75 --gt misc/nb201_gt.json --out <dir>
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.stats import kendalltau

ARCH_RE = re.compile(r"arch_str='([^']+)'")
PRED_TOKEN = {'cifar1042': 'cifar10', 'cifar10042': 'cifar100', 'imagenet42': 'imagenet16'}
TARGET = {'c10': 'cifar10', 'c100': 'cifar100', 'in16': 'imagenet16'}
GT_FIELD = {'cifar10': 'cifar10_test', 'cifar100': 'cifar100_test', 'imagenet16': 'imagenet16_test'}
LUT_STEPS = list(range(1, 12))
LIVE_STEPS = [1, 2, 3, 5, 7, 11, 17, 23]


def discover(root):
    runs = []
    for p in sorted(glob.glob(os.path.join(root, '*', '*', 'search-*', 'summary.json'))):
        kind, cell, run = p.split(os.sep)[-4:-1]
        m = re.match(r'search-(c10|c100|in16)_pred-(\w+?)_native', run)
        tgt, pred = TARGET[m.group(1)], PRED_TOKEN[m.group(2)]
        runs.append({'kind': kind, 'cell': cell, 'run': run, 'path': p, 'target': tgt, 'predictor': pred,
                     'label': f'{m.group(1)} in-dataset' if kind == 'in_dataset' else cell.replace('_to_', '->')})
    return runs


def unique_archs(summary):
    rows = {}
    for aid in sorted(summary['architectures'], key=int):
        a = summary['architectures'][aid]
        m = ARCH_RE.search(a.get('genotype', ''))
        if not m or m.group(1) in rows:
            continue
        fit = a.get('fitness') or {}
        rows[m.group(1)] = {'valid_acc': a.get('valid_acc'),
                            'live': {k: fit.get(f'nap2@{k}') for k in LIVE_STEPS}}
    return rows


def kt(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    xs, ys = zip(*pairs)
    return round(float(kendalltau(xs, ys).correlation), 4), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--lut', required=True)
    ap.add_argument('--gt', default='misc/nb201_gt.json')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = {a: v for a, v in json.load(open(args.gt)).items() if not a.startswith('_')}
    index = json.load(open(os.path.join(args.lut, 'arch_index.json')))
    tables = {}

    def table(pred, tgt, k):
        key = (pred, tgt, k)
        if key not in tables:
            tables[key] = json.load(open(os.path.join(args.lut, 'tables', f'trained_on_{pred}',
                                                      f'nap2_snap{k}_{tgt}.json')))
        return tables[key]

    long_rows, summary_rows, per_run = [], [], {}
    for r in discover(args.runs):
        rows = unique_archs(json.load(open(r['path'])))
        gtf = GT_FIELD[r['target']]
        side = {'_meta': {'source': 'nb201_lut_snap75', 'predictor': r['predictor'], 'target': r['target'],
                          'steps': LUT_STEPS, 'mini_batches_per_step': 75,
                          'leaked_split': 'train/val in arch_index.json = archs the predictor was fitted on'}}
        n_leak = 0
        for arch, row in rows.items():
            split = index.get(arch, {}).get('split', 'unknown')
            row['split'] = split
            n_leak += split in ('train', 'val')
            row['lut'] = {k: table(r['predictor'], r['target'], k).get(arch) for k in LUT_STEPS}
            row['gt'] = gt.get(arch, {}).get(gtf)
            side[arch] = {**{str(k): row['lut'][k] for k in LUT_STEPS}, 'split': split}
        rel = os.path.join(r['kind'], r['cell'], r['run'])
        os.makedirs(os.path.join(args.out, 'runs', rel), exist_ok=True)
        json.dump(side, open(os.path.join(args.out, 'runs', rel, 'nap2_snap75_lut.json'), 'w'))
        per_run[r['label']] = {'target': r['target'], 'predictor': r['predictor'], 'n': len(rows), 'n_leaked': n_leak}

        def emit(cadence, k, mb, score_of):
            clean = [a for a, row in rows.items() if row['split'] not in ('train', 'val')]
            kt_gt, n = kt([(score_of(row), row['gt']) for row in rows.values()])
            kt_gt_c, n_c = kt([(score_of(rows[a]), rows[a]['gt']) for a in clean])
            kt_va, _ = kt([(score_of(row), row['valid_acc']) for row in rows.values()])
            rec = {'run': r['label'], 'target': r['target'], 'predictor': r['predictor'], 'cadence': cadence,
                   'step': k, 'mini_batches': mb, 'kt_gt_all': kt_gt, 'kt_gt_clean': kt_gt_c,
                   'kt_valid20': kt_va, 'n': n, 'n_clean': n_c}
            long_rows.append(rec)
            return rec
        for k in LUT_STEPS:
            emit('75mb_lut', k, 75 * k, lambda row, k=k: row['lut'][k])
        for k in LIVE_STEPS:
            emit('100mb_live', k, 100 * k, lambda row, k=k: row['live'][k])

    fields = ['run', 'target', 'predictor', 'cadence', 'step', 'mini_batches', 'kt_gt_all', 'kt_gt_clean', 'kt_valid20', 'n', 'n_clean']
    with open(os.path.join(args.out, 'kt_by_step.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(long_rows)
    json.dump({'runs': per_run, 'rows': long_rows}, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)

    # README: matched-budget table (KT vs NB-201 GT, all visited archs; clean in parentheses)
    def cell(run, cad, k):
        for x in long_rows:
            if x['run'] == run and x['cadence'] == cad and x['step'] == k:
                return f"{x['kt_gt_all']:.3f} ({x['kt_gt_clean']:.3f})" if x['kt_gt_all'] is not None else 'n/a'
        return ''
    cols = [('75mb_lut', 1), ('100mb_live', 1), ('75mb_lut', 3), ('100mb_live', 2), ('75mb_lut', 4), ('100mb_live', 3),
            ('75mb_lut', 7), ('100mb_live', 5), ('75mb_lut', 11), ('100mb_live', 7), ('100mb_live', 11), ('100mb_live', 23)]
    hdr = ['run', 'n (leaked)'] + [f"{'LUT75' if c == '75mb_lut' else 'live100'} s{k} = {75 * k if c == '75mb_lut' else 100 * k} mb" for c, k in cols]
    L = ['# 75-mb lookup nap2 vs 100-mb live nap2 on the native runs (seed 42)', '',
         'Same architectures (unique archs of each run), Kendall tau vs NB-201 200-epoch test accuracy; in parentheses: '
         'excluding the 750 archs the 75-mb predictors were fitted on. LUT75 = `nb201_lut_snap75` tables '
         '(Alon\'s [w,g] single-dataset predictors, 75 mb/step); live100 = the run\'s own nap2 scores '
         '(Michael\'s predictors, 100 mb/step). Columns ordered by mini-batches.', '',
         '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for run, info in per_run.items():
        L.append('| ' + ' | '.join([run, f"{info['n']} ({info['n_leaked']})"] + [cell(run, c, k) for c, k in cols]) + ' |')
    L += ['', 'Full per-step table (incl. KT vs the run\'s own 20-epoch valid_acc): `kt_by_step.csv`; per-run side files: '
          '`runs/<kind>/<cell>/<run>/nap2_snap75_lut.json`; plot: `kt_vs_minibatches.png`.']
    open(os.path.join(args.out, 'README.md'), 'w').write('\n'.join(L) + '\n')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 10)); axes = axes.ravel()
    for ax, run in zip(axes, per_run):
        for cad, style, lab in [('75mb_lut', 's-', '75-mb lookup (Alon)'), ('100mb_live', 'o--', '100-mb live (Michael)')]:
            pts = [(x['mini_batches'], x['kt_gt_all']) for x in long_rows if x['run'] == run and x['cadence'] == cad and x['kt_gt_all'] is not None]
            ax.plot(*zip(*pts), style, ms=4, label=f'{lab}: vs NB-201 GT')
            pts = [(x['mini_batches'], x['kt_valid20']) for x in long_rows if x['run'] == run and x['cadence'] == cad and x['kt_valid20'] is not None]
            ax.plot(*zip(*pts), style, ms=3, alpha=0.4, label=f'{lab}: vs valid_acc 20ep')
        ax.set_title(f"{run} (n={per_run[run]['n']})", fontsize=9); ax.set_xlabel('mini-batches of partial training', fontsize=8)
        ax.set_ylabel('Kendall tau', fontsize=8); ax.set_xlim(0, 2400); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'kt_vs_minibatches.png'), dpi=120)
    print('wrote', sorted(os.listdir(args.out)))
    for run, info in per_run.items():
        print(f"{run:16} n={info['n']} leaked={info['n_leaked']}  LUT75 s7: {cell(run, '75mb_lut', 7)}  live100 s5: {cell(run, '100mb_live', 5)}  "
              f"LUT75 s11: {cell(run, '75mb_lut', 11)}  live100 s11: {cell(run, '100mb_live', 11)}")


if __name__ == '__main__':
    main()
