"""Seconds-normalization of native-cadence runs (Michael's protocol): LC methods at native epochs -> seconds -> nap2-snapshot
equivalents; paper-grid tables (rows = nap2 snapshot budgets 1,2,3,7,11,17,23);
KT vs in-run 20-epoch valid_acc and (CIFAR-10 targets) vs NB201 200-epoch GT."""
import json, os, re, sys, statistics, csv
from datetime import datetime
import numpy as np
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--runs', required=True, help='dir containing <tag>/summary.json + <tag>/log.txt per run')
_ap.add_argument('--gt', default='misc/nb201_gt.json')
_ap.add_argument('--out', required=True, help='output dir (normalized_report.json)')
_ap.add_argument('--run', action='append', default=[], help='tag:label:dataset (repeatable); default = the 9-run matrix')
_args = _ap.parse_args()
D = _args.runs
OUT = _args.out; os.makedirs(OUT, exist_ok=True)
LC = ['sotl', 'sotl_e', 'early_stop', 'lce_m', 'lc_pfn']; ZC = ['synflow', 'grad_norm', 'snip']
NAPB = [1, 2, 3, 5, 7, 11, 17, 23]; PAPER_ROWS = [1, 2, 3, 7, 11, 17, 23]
DEFAULT_RUNS = [('in_c10', 'C10 in-dataset', 'cifar10'), ('in_c100', 'C100 in-dataset', 'cifar100'),
                ('in_in16', 'IN16 in-dataset', 'ImageNet16-120'),
                ('x_c100_to_c10', 'C100->C10', 'cifar10'), ('x_in16_to_c10', 'IN16->C10', 'cifar10'),
                ('x_c10_to_c100', 'C10->C100', 'cifar100'), ('x_in16_to_c100', 'IN16->C100', 'cifar100'),
                ('x_c10_to_in16', 'C10->IN16', 'ImageNet16-120'), ('x_c100_to_in16', 'C100->IN16', 'ImageNet16-120')]
RUNS = [tuple(r.split(':')) for r in _args.run] if _args.run else \
       [r for r in DEFAULT_RUNS if os.path.exists(os.path.join(D, r[0], 'summary.json'))]
GT = json.load(open(_args.gt))
GTF = {'cifar10': 'cifar10_test', 'cifar100': 'cifar100_test', 'ImageNet16-120': 'imagenet16_test'}
ARCH_RE = re.compile(r"arch_str='([^']+)'")
TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) ')

def ts(line):
    m = TS.match(line); return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f') if m else None

def timings(log_path):
    """Per-arch nap2 query time (Architecture line -> nap2 pred_acc line) and per-epoch train times."""
    nap2_t, ep_t, val_t = [], [], []
    t_arch = None; last_train = None
    for line in open(log_path, errors='replace'):
        if 'Architecture = ' in line: t_arch = ts(line)
        elif 'nap2 pred_acc' in line and t_arch is not None:
            nap2_t.append((ts(line) - t_arch).total_seconds()); t_arch = None
        elif ' train_time ' in line:
            ep_t.append(float(line.rsplit('train_time', 1)[1].strip().rstrip('s'))); last_train = ts(line)
        elif re.search(r'epoch \d+ val_acc', line) and last_train is not None:
            val_t.append((ts(line) - last_train).total_seconds()); last_train = None
    return nap2_t, ep_t, val_t

def kt_vs(preds_by_key, truth):
    out = {}
    for k, d in preds_by_key.items():
        common = [a for a in d if a in truth and d[a] is not None]
        if len(common) < 3: out[k] = None; continue
        x = [d[a] for a in common]; y = [truth[a] for a in common]
        tau = kendalltau(x, y)[0]; out[k] = None if tau != tau else float(tau)
    return out

def per_key_scores(summary):
    """{key: {arch_str: score}} deduped by arch_str (first occurrence), plus valid_acc truth."""
    keys, truth = {}, {}
    for aid, a in summary['architectures'].items():
        m = ARCH_RE.search(a.get('genotype', ''))
        if not m: continue
        s = m.group(1)
        if s in truth: continue
        truth[s] = a['valid_acc']
        for k, v in (a.get('fitness') or {}).items(): keys.setdefault(k, {})[s] = v
        if a.get('pred_acc') is not None: keys.setdefault('nap2', {})[s] = a['pred_acc']
    return keys, truth

def interp(points, x):
    """Linear interpolation on sorted (x, y); None outside [x_min, x_max]."""
    pts = [p for p in points if p[1] is not None]
    if not pts or x < pts[0][0] - 1e-9 or x > pts[-1][0] + 1e-9: return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 - 1e-9 <= x <= x1 + 1e-9:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1] if abs(x - pts[-1][0]) < 1e-9 else None

report = {}
for tag, label, ds in RUNS:
    summ = json.load(open(f'{D}/{tag}/summary.json'))
    nap2_t, ep_t, val_t = timings(f'{D}/{tag}/log.txt')
    T_nap2 = statistics.median(nap2_t); T_snap = T_nap2 / 23.0
    T_epoch = statistics.median(ep_t); T_val = statistics.median(val_t) if val_t else float('nan')
    r = T_epoch / T_snap                       # snapshot-equivalents per epoch (seconds-based)
    keys, truth = per_key_scores(summ)
    fm = summ['fitness_metrics']
    kt_in = {k: (fm[k]['kendall_tau'] if k in fm else None) for k in keys}
    kt_in['nap2@23'] = kt_in.get('nap2@23') or summ['metrics']['kendall_tau']
    gtruth = {s: GT[s][GTF[ds]] for s in truth if s in GT}
    gt_kt = kt_vs(keys, gtruth)
    print(f'   NB201 GT ({GTF[ds]}): matched {len(gtruth)}/{len(truth)} archs')
    eb = sorted({int(m.group(1)) for k in fm for m in [re.match(r'.+@e(\d+)$', k)] if m})

    def grid_table(ktsrc):
        rows = []
        for m in LC:
            native = [(K * r, ktsrc.get(f'{m}@e{K}')) for K in eb]
            row = {'method': m}
            for k in PAPER_ROWS:
                v = interp(native, k)
                row[f'@{k}'] = None if v is None else round(v, 4)
                row[f'@{k}_flag'] = ('n/a<1ep' if k < r - 1e-9 else ('exact' if any(abs(K * r - k) < 0.02 for K in eb) else 'interp'))
            rows.append(row)
        row = {'method': 'nap2'}
        for k in PAPER_ROWS:
            v = ktsrc.get(f'nap2@{k}'); row[f'@{k}'] = None if v is None else round(v, 4); row[f'@{k}_flag'] = 'native'
        rows.append(row)
        for m in ZC:
            v = ktsrc.get(m); row = {'method': m}
            for k in PAPER_ROWS: row[f'@{k}'] = None if v is None else round(v, 4); row[f'@{k}_flag'] = 'init'
            rows.append(row)
        return rows

    native_pts = {m: [(K, round(K * r, 2), round(K * T_epoch, 1), kt_in.get(f'{m}@e{K}')) for K in eb] for m in LC}
    report[tag] = dict(label=label, dataset=ds, n=len(truth), eb=eb,
                       T_nap2_23=round(T_nap2, 1), T_snap=round(T_snap, 2), T_epoch=round(T_epoch, 2),
                       T_val=round(T_val, 2), r=round(r, 3),
                       grid_valid=grid_table(kt_in), grid_nb201=(grid_table(gt_kt) if gt_kt else None),
                       native_valid=native_pts,
                       nap2_native=[(k, round(k * T_snap, 1), kt_in.get(f'nap2@{k}')) for k in NAPB],
                       zc_valid={m: kt_in.get(m) for m in ZC},
                       nb201_nap2=({k: gt_kt.get(f'nap2@{k}') for k in NAPB} if gt_kt else None),
                       nb201_zc=({m: gt_kt.get(m) for m in ZC} if gt_kt else None),
                       nb201_native={m: [(K, gt_kt.get(f'{m}@e{K}')) for K in eb] for m in LC} if gt_kt else None)
    print(f"{label:>16}: n={len(truth)} T_epoch={T_epoch:.2f}s T_snap={T_snap:.2f}s (nap2@23={T_nap2:.0f}s) "
          f"val={T_val:.2f}s  r={r:.3f} snap-eq/epoch  epochs={eb[0]}..{eb[-1]}")
json.dump(report, open(f'{OUT}/normalized_report.json', 'w'), indent=1)
print('wrote', f'{OUT}/normalized_report.json')
