#!/usr/bin/env python3
"""Why do learning-curve proxies degrade on IN16? Early-signal diagnostic from native-run summaries.

For every dataset, pools the architectures visited by the native in-dataset runs (all seeds given),
and per training epoch e computes
  * progress(e)  = (mean val acc at e - chance) / (mean GT - chance)   how far above chance the curve is
  * snr(e)       = (mean val acc at e - chance) / std across archs     signal above chance vs spread
  * kt_es(e)     = Kendall tau(early_stop@e, GT)
  * kt_sotl(e)   = Kendall tau(sotl_e@e, GT)
plus the in-run nap2@k KT as a reference. Writes a CSV and a 3-panel PNG.

    .venv/bin/python scripts/lc_signal_diagnostic.py --summaries DIR --gt data/nb201_nats_gt.json --out OUTDIR
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import kendalltau

DS = {'c10': ('cifar10_test', 10), 'c100': ('cifar100_test', 100), 'in16': ('imagenet16_test', 120)}
LABEL = {'c10': 'CIFAR-10', 'c100': 'CIFAR-100', 'in16': 'ImageNet16-120'}


def load(summaries, ds):
    archs = {}
    for f in sorted(glob.glob(os.path.join(summaries, f'native_{ds}_seed*.json'))):
        for a in json.load(open(f))['architectures'].values():
            s = re.search(r"arch_str='([^']+)'", a['genotype']).group(1)
            archs.setdefault(s, a)  # first seed wins; duplicates across seeds are the same arch
    return archs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', required=True)
    ap.add_argument('--gt', default='data/nb201_nats_gt.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gt_all = json.load(open(args.gt))
    rows, curves = [], {}
    for ds, (gkey, ncls) in DS.items():
        archs = load(args.summaries, ds)
        keys = [s for s in archs if s in gt_all]
        gt = np.array([gt_all[s][gkey] for s in keys])
        chance = 100.0 / ncls
        nap2 = set.intersection(*[{k for k in archs[s]['fitness'] if k.startswith('nap2@')} for s in keys])
        kt_nap2 = max(kendalltau([archs[s]['fitness'][k] for s in keys], gt)[0] for k in nap2)
        c = {'progress': [], 'snr': [], 'kt_es': [], 'kt_sotl': [], 'kt_nap2': kt_nap2, 'n': len(keys),
             'gt_mean': gt.mean(), 'gt_std': gt.std(), 'chance': chance}
        for e in range(1, args.epochs + 1):
            es = np.array([archs[s]['fitness'][f'early_stop@e{e}'] for s in keys]) * 100
            so = np.array([archs[s]['fitness'][f'sotl_e@e{e}'] for s in keys])
            c['progress'].append((es.mean() - chance) / (gt.mean() - chance))
            c['snr'].append((es.mean() - chance) / es.std())
            c['kt_es'].append(kendalltau(es, gt)[0])
            c['kt_sotl'].append(kendalltau(so, gt)[0])
            rows.append((ds, e, len(keys), chance, es.mean(), es.std(), c['progress'][-1], c['snr'][-1],
                         c['kt_es'][-1], c['kt_sotl'][-1], kt_nap2))
        curves[ds] = c
    with open(os.path.join(args.out, 'lc_signal_diagnostic.csv'), 'w') as f:
        f.write('dataset,epoch,n_archs,chance_acc,mean_val_acc,std_val_acc,progress_to_gt,snr_above_chance,kt_early_stop,kt_sotl_e,kt_nap2_best_inrun\n')
        for r in rows:
            f.write(','.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in r) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = np.arange(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = {'c10': 'tab:blue', 'c100': 'tab:orange', 'in16': 'tab:red'}
    for ds, c in curves.items():
        ax[0].plot(ep, c['progress'], '-o', ms=3, color=cols[ds], label=f"{LABEL[ds]} (chance {c['chance']:.1f}%)")
        ax[1].plot(ep, c['snr'], '-o', ms=3, color=cols[ds], label=LABEL[ds])
        ax[2].plot(ep, c['kt_es'], '-o', ms=3, color=cols[ds], label=f'{LABEL[ds]} early_stop')
        ax[2].plot(ep, c['kt_sotl'], '--', color=cols[ds], label=f'{LABEL[ds]} sotl_e')
        ax[2].axhline(c['kt_nap2'], color=cols[ds], ls=':', lw=1.2)
    ax[2].plot([], [], 'k:', label='nap2 (dotted, in-run best k)')
    ax[0].set_title('Val-acc progress above chance, as fraction of final GT'); ax[0].set_ylim(0, 1.05)
    ax[1].set_title('(mean val acc - chance) / std across archs')
    ax[2].set_title('Kendall tau vs 200-epoch GT'); ax[2].set_ylim(-0.1, 0.9)
    for a in ax:
        a.set_xlabel('training epoch'); a.grid(alpha=.3); a.legend(fontsize=8)
    n = {ds: c['n'] for ds, c in curves.items()}
    fig.suptitle(f'Early-training signal vs dataset (native in-dataset runs, pooled seeds; archs {n})', fontsize=11)
    fig.tight_layout()
    png = os.path.join(args.out, 'lc_signal_diagnostic.png')
    fig.savefig(png, dpi=140)
    print('wrote', png)
    for ds, c in curves.items():
        print(f"{ds:5s} n={c['n']:4d} chance={c['chance']:.2f} GT mean={c['gt_mean']:.1f} std={c['gt_std']:.1f} | "
              f"e1 prog={c['progress'][0]:.2f} snr={c['snr'][0]:.2f} ktES={c['kt_es'][0]:.2f} ktSoTL={c['kt_sotl'][0]:.2f} | "
              f"e5 prog={c['progress'][4]:.2f} snr={c['snr'][4]:.2f} ktES={c['kt_es'][4]:.2f} ktSoTL={c['kt_sotl'][4]:.2f} | "
              f"e20 ktES={c['kt_es'][-1]:.2f} ktSoTL={c['kt_sotl'][-1]:.2f} | nap2 {c['kt_nap2']:.2f}")


if __name__ == '__main__':
    main()
