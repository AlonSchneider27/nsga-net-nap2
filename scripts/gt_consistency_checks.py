#!/usr/bin/env python3
"""Ground-truth consistency checks for the native (valid_acc-guided) runs.

Check 1 - over each run's UNIQUE architectures, rank correlations (Kendall tau,
Spearman) among: our 20-epoch valid_acc, NB-201's own accuracy after 20 and
after 200 epochs (valid and test splits, from misc/nb201_epoch_gt.json), the
single-trial 200-epoch GT used so far (misc/nb201_gt.json), and nap2 at every
budget. Also pooled per target dataset (union of the 3 runs, nap2 excluded
because the predictor differs per run) and over the full 15,625-arch space
(NB-201 epoch-20 vs epoch-200 only).

Check 2 - our 20-epoch valid_acc vs NB-201's reported accuracies (same-epoch 20
and final 200): KT / Spearman / Pearson / MAE / bias / top-10 overlap + scatter.

Usage (from repo root)::

    python scripts/gt_consistency_checks.py --runs <research/native/seed_42> \
        --epoch-gt misc/nb201_epoch_gt.json --gt misc/nb201_gt.json --out <dir>

``--runs`` accepts the research-tree layout (in_dataset/<ds>/search-*/,
cross_dataset/<src>_to_<tgt>/search-*/) or a flat <tag>/summary.json layout.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ARCH_RE = re.compile(r"arch_str='([^']+)'")
NAP2_BUDGETS = [1, 2, 3, 5, 7, 11, 17, 23]
RUNS = [  # tag, label, target dataset, predictor source
    ('in_c10', 'C10 in-dataset', 'cifar10', 'c10'),
    ('in_c100', 'C100 in-dataset', 'cifar100', 'c100'),
    ('in_in16', 'IN16 in-dataset', 'ImageNet16-120', 'in16'),
    ('x_c100_to_c10', 'C100->C10', 'cifar10', 'c100'),
    ('x_in16_to_c10', 'IN16->C10', 'cifar10', 'in16'),
    ('x_c10_to_c100', 'C10->C100', 'cifar100', 'c10'),
    ('x_in16_to_c100', 'IN16->C100', 'cifar100', 'in16'),
    ('x_c10_to_in16', 'C10->IN16', 'ImageNet16-120', 'c10'),
    ('x_c100_to_in16', 'C100->IN16', 'ImageNet16-120', 'c100'),
]
DS_SHORT = {'cifar10': 'c10', 'cifar100': 'c100', 'ImageNet16-120': 'in16'}
GT200_FIELD = {'cifar10': 'cifar10_test', 'cifar100': 'cifar100_test',
               'ImageNet16-120': 'imagenet16_test'}
# NB-201 epoch-GT field per (target dataset, column). The .pth evaluates x-valid /
# x-test at EVERY epoch only for cifar10-valid (x-valid) and cifar10 (ori-test);
# for cifar100 / ImageNet16-120 the per-epoch series is ori-test (= x-valid u
# x-test, the full 10k / 6k held-out set), while x-valid / x-test exist only at
# epoch 200. So for those two datasets the epoch-20 column is the ori-test
# ('valtest') accuracy and nb_test20 is unavailable.
EPOCH_FIELD = {
    'cifar10': {'nb_valid20': 'c10v_valid_ep20', 'nb_valid200': 'c10v_valid_ep200',
                'nb_test20': 'c10_test_ep20', 'nb_test200': 'c10_test_ep200'},
    'cifar100': {'nb_valid20': 'c100_valtest_ep20', 'nb_valid200': 'c100_valid_ep200',
                 'nb_test20': None, 'nb_test200': 'c100_test_ep200'},
    'ImageNet16-120': {'nb_valid20': 'in16_valtest_ep20', 'nb_valid200': 'in16_valid_ep200',
                       'nb_test20': None, 'nb_test200': 'in16_test_ep200'},
}
NB_COLS = ['nb_valid20', 'nb_test20', 'nb_valid200', 'nb_test200']
BASE_COLS = ['ours20'] + NB_COLS + ['gt200_naslib']
NAP2_COLS = [f'nap2@{k}' for k in NAP2_BUDGETS]


# ------------------------------------------------------------------ loading
def discover_runs(root):
    out = []
    for tag, label, ds, src in RUNS:
        cands = [os.path.join(root, tag, 'summary.json')]
        if tag.startswith('in_'):
            cands += glob.glob(os.path.join(root, 'in_dataset', tag[3:], 'search-*', 'summary.json'))
        else:
            pair = tag[2:]
            cands += glob.glob(os.path.join(root, 'cross_dataset', pair, 'search-*', 'summary.json'))
        found = sorted(p for p in cands if os.path.exists(p))
        if not found:
            print(f'WARNING: no summary.json for {tag} under {root}', file=sys.stderr)
            continue
        out.append((tag, label, ds, src, found[-1]))
    return out


def load_run(path):
    """arch_str -> row (first occurrence wins; accuracies in percent)."""
    s = json.load(open(path))
    rows, dup = {}, 0
    for aid in sorted(s['architectures'], key=int):
        a = s['architectures'][aid]
        m = ARCH_RE.search(a.get('genotype', ''))
        if not m:
            continue
        arch = m.group(1)
        if arch in rows:
            dup += 1
            rows[arch]['n_evals'] += 1
            continue
        fit = a.get('fitness') or {}
        row = {'arch_id': int(aid), 'n_evals': 1,
               'ours20': 100.0 * a['valid_acc'] if a.get('valid_acc') is not None else None,
               'flops': a.get('flops'), 'params': a.get('param_size_mb')}
        for k in NAP2_BUDGETS:
            v = fit.get(f'nap2@{k}')
            if v is None and k == max(NAP2_BUDGETS):
                v = a.get('pred_acc')
            row[f'nap2@{k}'] = 100.0 * v if v is not None else None
        rows[arch] = row
    return rows, dup


def attach_gt(rows, ds, epoch_gt, gt):
    fields = EPOCH_FIELD[ds]
    missing = 0
    for arch, r in rows.items():
        g = gt.get(arch)
        r['gt200_naslib'] = g[GT200_FIELD[ds]] if g else None
        e = epoch_gt.get(arch) if epoch_gt else None
        if e is None:
            missing += (g is None)
            for c in NB_COLS:
                r[c] = None
            continue
        for c in NB_COLS:
            r[c] = e.get(fields[c]) if fields[c] else None
    return missing


# ------------------------------------------------------------------ stats
def _pairs(rows, x, y):
    xs, ys = [], []
    for r in rows.values():
        a, b = r.get(x), r.get(y)
        if a is None or b is None:
            continue
        xs.append(float(a)); ys.append(float(b))
    return np.array(xs), np.array(ys)


def corr(rows, x, y):
    xs, ys = _pairs(rows, x, y)
    n = len(xs)
    if n < 3 or np.ptp(xs) == 0 or np.ptp(ys) == 0:
        return {'n': n, 'kt': None, 'rho': None, 'r': None}
    kt = stats.kendalltau(xs, ys).correlation
    rho = stats.spearmanr(xs, ys).correlation
    r = stats.pearsonr(xs, ys)[0]
    f = lambda v: None if v is None or np.isnan(v) else round(float(v), 4)
    return {'n': n, 'kt': f(kt), 'rho': f(rho), 'r': f(r)}


def agreement(rows, x, y, top_k=10):
    c = corr(rows, x, y)
    xs, ys = _pairs(rows, x, y)
    if len(xs) < 3:
        return c
    d = xs - ys
    top_x = set(np.argsort(-xs)[:top_k]); top_y = set(np.argsort(-ys)[:top_k])
    c.update({'mae_pp': round(float(np.mean(np.abs(d))), 3),
              'bias_pp': round(float(np.mean(d)), 3),
              'sd_diff_pp': round(float(np.std(d)), 3),
              'top10_overlap': len(top_x & top_y) / float(top_k)})
    return c


def matrix(rows, cols, stat):
    m = {}
    for a in cols:
        m[a] = {}
        for b in cols:
            m[a][b] = 1.0 if a == b else corr(rows, a, b)[stat]
    return m


# ------------------------------------------------------------------ outputs
def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def fmt(v):
    return '' if v is None else (f'{v:.3f}' if isinstance(v, float) else str(v))


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        out.append('| ' + ' | '.join(fmt(v) for v in r) + ' |')
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--epoch-gt', default='')
    p.add_argument('--gt', default='misc/nb201_gt.json')
    p.add_argument('--out', required=True)
    p.add_argument('--no-xlsx', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    gt = {a: v for a, v in json.load(open(args.gt)).items() if not a.startswith('_')}
    epoch_gt = None
    if args.epoch_gt:
        epoch_gt = {a: v for a, v in json.load(open(args.epoch_gt)).items() if not a.startswith('_')}
    have_epoch = epoch_gt is not None
    cols = BASE_COLS if have_epoch else ['ours20', 'gt200_naslib']
    all_cols = cols + NAP2_COLS

    runs = discover_runs(args.runs)
    data = {}
    per_arch, long_rows, check1, check2, budgets, run_info = [], [], [], [], [], []
    for tag, label, ds, src, path in runs:
        rows, dup = load_run(path)
        missing = attach_gt(rows, ds, epoch_gt, gt)
        data[tag] = (label, ds, src, rows)
        run_info.append({'run': tag, 'label': label, 'dataset': ds, 'predictor': src,
                         'n_unique': len(rows), 'n_dup_evals': dup, 'n_missing_gt': missing})
        for arch, r in rows.items():
            per_arch.append({'run': tag, 'dataset': ds, 'predictor': src, 'arch_str': arch, **r})
        km = matrix(rows, all_cols, 'kt'); rm = matrix(rows, all_cols, 'rho')
        with open(os.path.join(args.out, f'kt_matrix_{tag}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow([''] + all_cols)
            for a in all_cols:
                w.writerow([a] + [fmt(km[a][b]) for b in all_cols])
        for i, a in enumerate(all_cols):
            for b in all_cols[i + 1:]:
                c = corr(rows, a, b)
                long_rows.append({'scope': 'run', 'run': tag, 'dataset': ds, 'x': a, 'y': b, **c})
        c1 = {'run': tag, 'dataset': ds, 'n': len(rows)}
        for a, b in [('ours20', 'nb_valid20'), ('ours20', 'nb_test20'), ('ours20', 'nb_valid200'),
                     ('ours20', 'nb_test200'), ('ours20', 'gt200_naslib'),
                     ('nb_valid20', 'nb_valid200'), ('nb_test20', 'nb_test200'),
                     ('nb_valid20', 'nb_test200'), ('gt200_naslib', 'nb_test200'),
                     ('nap2@23', 'ours20'), ('nap2@23', 'nb_valid20'), ('nap2@23', 'nb_test20'),
                     ('nap2@23', 'nb_test200'), ('nap2@23', 'gt200_naslib')]:
            if a in all_cols and b in all_cols:
                c1[f'kt_{a}_vs_{b}'] = km[a][b]
        check1.append(c1)
        for tgt in cols:
            budgets.append({'run': tag, 'dataset': ds, 'target': tgt,
                            **{f'@{k}': km[f'nap2@{k}'][tgt] for k in NAP2_BUDGETS}})
        for tgt in (NB_COLS if have_epoch else []) + ['gt200_naslib']:
            check2.append({'run': tag, 'dataset': ds, 'comparison': tgt, **agreement(rows, 'ours20', tgt)})

    # pooled per dataset (nap2 excluded), ours20 averaged over runs where present
    pooled = {}
    for tag, (label, ds, src, rows) in data.items():
        pr = pooled.setdefault(ds, {})
        for arch, r in rows.items():
            q = pr.setdefault(arch, {**{c: r.get(c) for c in cols}, '_ours': []})
            if r.get('ours20') is not None:
                q['_ours'].append(r['ours20'])
    for ds, pr in pooled.items():
        for q in pr.values():
            q['ours20'] = float(np.mean(q['_ours'])) if q['_ours'] else None
        km = matrix(pr, cols, 'kt')
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                long_rows.append({'scope': 'pooled', 'run': f'pooled_{DS_SHORT[ds]}', 'dataset': ds,
                                  'x': a, 'y': b, **corr(pr, a, b)})
        c1 = {'run': f'pooled_{DS_SHORT[ds]}', 'dataset': ds, 'n': len(pr)}
        for a, b in [('ours20', 'nb_valid20'), ('ours20', 'nb_test20'), ('ours20', 'nb_valid200'),
                     ('ours20', 'nb_test200'), ('ours20', 'gt200_naslib'), ('nb_valid20', 'nb_valid200'),
                     ('nb_test20', 'nb_test200'), ('nb_valid20', 'nb_test200'), ('gt200_naslib', 'nb_test200')]:
            if a in cols and b in cols:
                c1[f'kt_{a}_vs_{b}'] = km[a][b]
        check1.append(c1)
        for tgt in (NB_COLS if have_epoch else []) + ['gt200_naslib']:
            check2.append({'run': f'pooled_{DS_SHORT[ds]}', 'dataset': ds, 'comparison': tgt,
                           **agreement(pr, 'ours20', tgt)})

    # full space: NB-201 epoch-20 vs epoch-200 over all 15,625 archs
    full = []
    if have_epoch:
        for ds, fields in EPOCH_FIELD.items():
            fr = {a: {**{c: (e.get(fields[c]) if fields[c] else None) for c in NB_COLS},
                      'gt200_naslib': gt.get(a, {}).get(GT200_FIELD[ds])}
                  for a, e in epoch_gt.items()}
            row = {'dataset': ds, 'n': len(fr)}
            for a, b in [('nb_valid20', 'nb_valid200'), ('nb_test20', 'nb_test200'),
                         ('nb_valid20', 'nb_test200'), ('nb_valid200', 'nb_test200'),
                         ('gt200_naslib', 'nb_test200')]:
                c = corr(fr, a, b)
                row[f'kt_{a}_vs_{b}'] = c['kt']; row[f'rho_{a}_vs_{b}'] = c['rho']
                long_rows.append({'scope': 'full_space', 'run': 'full', 'dataset': ds, 'x': a, 'y': b, **c})
            full.append(row)

    # ---- write
    pa_fields = ['run', 'dataset', 'predictor', 'arch_str', 'arch_id', 'n_evals'] + all_cols + ['flops', 'params']
    write_csv(os.path.join(args.out, 'per_arch_all.csv'), per_arch, pa_fields)
    write_csv(os.path.join(args.out, 'correlations_long.csv'), long_rows,
              ['scope', 'run', 'dataset', 'x', 'y', 'n', 'kt', 'rho', 'r'])
    c1_fields = sorted({k for r in check1 for k in r}, key=lambda k: (k not in ('run', 'dataset', 'n'), k))
    write_csv(os.path.join(args.out, 'check1_summary.csv'), check1, c1_fields)
    write_csv(os.path.join(args.out, 'check1_nap2_budgets.csv'), budgets,
              ['run', 'dataset', 'target'] + [f'@{k}' for k in NAP2_BUDGETS])
    write_csv(os.path.join(args.out, 'check2_summary.csv'), check2,
              ['run', 'dataset', 'comparison', 'n', 'kt', 'rho', 'r', 'mae_pp', 'bias_pp', 'sd_diff_pp', 'top10_overlap'])
    if full:
        write_csv(os.path.join(args.out, 'full_space_nb201.csv'), full, list(full[0].keys()))
    json.dump({'runs': run_info, 'check1': check1, 'check2': check2, 'nap2_budgets': budgets,
               'full_space': full, 'columns': all_cols, 'have_epoch_gt': have_epoch},
              open(os.path.join(args.out, 'report.json'), 'w'), indent=1)

    if not args.no_plots:
        make_plots(data, cols, args.out, have_epoch)
    if not args.no_xlsx:
        try:
            make_xlsx(data, all_cols, check1, c1_fields, check2, budgets, full, args.out)
        except ImportError:
            print('openpyxl not available: xlsx skipped', file=sys.stderr)
    write_readme(run_info, check1, check2, budgets, full, have_epoch, args.out)
    print('wrote', sorted(os.listdir(args.out)))


def make_plots(data, cols, out, have_epoch):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tgts = (NB_COLS if have_epoch else []) + ['gt200_naslib']
    tags = list(data)
    fig, axes = plt.subplots(len(tags), len(tgts), figsize=(3.6 * len(tgts), 3.2 * len(tags)), squeeze=False)
    for i, tag in enumerate(tags):
        label, ds, src, rows = data[tag]
        for j, tgt in enumerate(tgts):
            ax = axes[i][j]; xs, ys = _pairs(rows, tgt, 'ours20')
            if len(xs):
                ax.scatter(xs, ys, s=6, alpha=0.5)
                lo, hi = min(xs.min(), ys.min()), max(xs.max(), ys.max())
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7)
                c = agreement(rows, 'ours20', tgt)
                ax.set_title(f"{label}\nvs {tgt}: KT {fmt(c['kt'])} MAE {fmt(c.get('mae_pp'))}", fontsize=8)
            ax.set_xlabel(tgt, fontsize=7); ax.set_ylabel('ours valid_acc @20ep', fontsize=7); ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'scatter_check2_grid.png'), dpi=120); plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10)); axes = axes.ravel()
    for ax, tag in zip(axes, tags):
        label, ds, src, rows = data[tag]
        for tgt in cols:
            ys = [corr(rows, f'nap2@{k}', tgt)['kt'] for k in NAP2_BUDGETS]
            ax.plot(NAP2_BUDGETS, [np.nan if y is None else y for y in ys], marker='o', ms=3, label=tgt)
        ax.set_title(label, fontsize=9); ax.set_xlabel('nap2 budget (snapshots)', fontsize=8)
        ax.set_ylabel('KT(nap2@k, target)', fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'nap2_budget_curves.png'), dpi=120); plt.close(fig)


def make_xlsx(data, all_cols, check1, c1_fields, check2, budgets, full, out):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    HF = PatternFill('solid', fgColor='305496'); HFONT = Font(color='FFFFFF', bold=True, size=10)
    wb = Workbook(); wb.remove(wb.active)

    def sheet(name, fields, rows):
        ws = wb.create_sheet(name)
        for j, h in enumerate(fields, 1):
            c = ws.cell(1, j, h); c.fill = HF; c.font = HFONT; c.alignment = Alignment(horizontal='center')
        for i, r in enumerate(rows, 2):
            for j, h in enumerate(fields, 1):
                ws.cell(i, j, r.get(h))
        ws.freeze_panes = 'B2'
    sheet('check1_KT', c1_fields, check1)
    sheet('check2_ours_vs_nb201', ['run', 'dataset', 'comparison', 'n', 'kt', 'rho', 'r', 'mae_pp', 'bias_pp', 'sd_diff_pp', 'top10_overlap'], check2)
    sheet('nap2_budgets_KT', ['run', 'dataset', 'target'] + [f'@{k}' for k in NAP2_BUDGETS], budgets)
    if full:
        sheet('full_space_nb201', list(full[0].keys()), full)
    for tag, (label, ds, src, rows) in data.items():
        ws = wb.create_sheet(f'M_{tag}'[:31])
        for r0, (title, stat) in enumerate([('Kendall tau', 'kt'), ('Spearman rho', 'rho')]):
            base = r0 * (len(all_cols) + 4)
            ws.cell(base + 1, 1, f'{label} ({ds}, n={len(rows)}) — {title}').font = Font(bold=True)
            m = matrix(rows, all_cols, stat)
            for j, h in enumerate(all_cols, 2):
                c = ws.cell(base + 2, j, h); c.fill = HF; c.font = HFONT
            for i, a in enumerate(all_cols, 3):
                ws.cell(base + i, 1, a).font = Font(bold=True)
                for j, b in enumerate(all_cols, 2):
                    ws.cell(base + i, j, m[a][b])
    wb.save(os.path.join(out, 'gt_consistency.xlsx'))


def write_readme(run_info, check1, check2, budgets, full, have_epoch, out):
    L = ['# NB-201 ground-truth consistency checks — native runs, seed 42', '',
         'Unique architectures per run (first evaluation kept); accuracies in percent. '
         'ours20 = our 20-epoch valid_acc (NB-201 recipe, T_max=200, 40k/10k CIFAR split, 151.7k/6k IN16). '
         'nb_valid20/200 = NB-201 x-valid accuracy after 20 / 200 of its own 200 epochs (cifar10: the cifar10-valid 25k/25k run); '
         'nb_test20/200 = NB-201 test accuracy (cifar10: ori-test of the 50k run; cifar100/IN16: x-test). '
         'gt200_naslib = single-trial 200-epoch test accuracy used in all previous tables (misc/nb201_gt.json). '
         'nap2@k = nap2 prediction after k snapshots (100 mb each).', '']
    if not have_epoch:
        L.append('**NOTE: run without --epoch-gt; NB-201 epoch-20/200 columns absent.**\n')
    L += ['## Runs', md_table(['run', 'dataset', 'predictor', 'n_unique', 'dup_evals', 'missing_gt'],
                             [[r['run'], r['dataset'], r['predictor'], r['n_unique'], r['n_dup_evals'], r['n_missing_gt']] for r in run_info]), '']
    keys = [k for k in check1[0] if k.startswith('kt_')]
    L += ['## Check 1 — Kendall tau (per run; pooled = union of the 3 runs per target, nap2 excluded)',
          md_table(['run', 'n'] + [k[3:] for k in keys], [[r['run'], r['n']] + [r.get(k) for k in keys] for r in check1]), '']
    L += ['## nap2 budgets — KT(nap2@k, target)',
          md_table(['run', 'target'] + [f'@{k}' for k in NAP2_BUDGETS],
                   [[b['run'], b['target']] + [b[f'@{k}'] for k in NAP2_BUDGETS] for b in budgets]), '']
    L += ['## Check 2 — our 20-epoch valid_acc vs NB-201 reported accuracy',
          md_table(['run', 'vs', 'n', 'KT', 'rho', 'pearson', 'MAE pp', 'bias pp', 'top10 overlap'],
                   [[r['run'], r['comparison'], r['n'], r['kt'], r['rho'], r['r'], r.get('mae_pp'), r.get('bias_pp'), r.get('top10_overlap')] for r in check2]), '']
    if full:
        fk = [k for k in full[0] if k.startswith('kt_')]
        L += ['## Full space (15,625 archs) — NB-201 epoch 20 vs 200',
              md_table(['dataset'] + [k[3:] for k in fk], [[r['dataset']] + [r[k] for k in fk] for r in full]), '']
    L += ['## Findings', '', '(to be filled after reading the numbers)', '',
          'Files: per_arch_all.csv, correlations_long.csv, check1_summary.csv, check1_nap2_budgets.csv, check2_summary.csv, '
          'full_space_nb201.csv, kt_matrix_<run>.csv, gt_consistency.xlsx, scatter_check2_grid.png, nap2_budget_curves.png, report.json']
    open(os.path.join(out, 'README.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
