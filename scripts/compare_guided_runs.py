#!/usr/bin/env python3
"""Compare method-guided GA runs against NB-201 ground truth.

Takes N finished guided-run dirs (each with summary.json + final_pop.json),
scores every visited architecture with the reported NB-201 accuracies
(misc/nb201_cifar10_gt.json), and emits a cross-method comparison: best/top-k
found, regret vs the GT optimum and vs a random-search baseline, convergence
curves, guidance fidelity (KT of the method's own score vs GT), overlap
between methods, and operation-composition profiles.

Usage from project root::

    python scripts/compare_guided_runs.py --runs <parent-dir-of-search-*> \
        --gt misc/nb201_cifar10_gt.json --out-prefix <prefix>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_top_archs import extract_arch_str, resolve_rank_key, _score_of

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP_RE = re.compile(r'\|([a-z_0-9]+)~')


def load_gt(path, key='cifar10_test'):
    raw = json.load(open(path))
    return {a: v[key] for a, v in raw.items() if not a.startswith('_')}


def load_run(run_dir):
    """One guided run -> dict(method, visited=[(arch_str, score)] in eval order,
    front=[arch_str])."""
    summary = json.load(open(os.path.join(run_dir, 'summary.json')))
    method = summary.get('objective_method') or 'valid_acc'
    archs = summary['architectures']
    kind, key = resolve_rank_key(archs, method) if method != 'valid_acc' \
        else ('attr', 'valid_acc')
    visited = []
    for arch_id in sorted(archs, key=int):
        a = archs[arch_id]
        visited.append((extract_arch_str(a['genotype']), _score_of(a, kind, key)))
    front = []
    fp_path = os.path.join(run_dir, 'final_pop.json')
    if os.path.exists(fp_path):
        fp = json.load(open(fp_path))
        front = [e.get('arch_str') or extract_arch_str(e['genotype'])
                 for e in fp['population'] if e.get('on_pareto_front')]
    return {'method': method, 'visited': visited, 'front': front}


def dedupe_keep_first(visited):
    seen, out = set(), []
    for arch, score in visited:
        if arch not in seen:
            seen.add(arch)
            out.append((arch, score))
    return out


def convergence_curve(visited, gt):
    """Best GT-so-far at each evaluation index (including duplicates)."""
    best, curve = float('-inf'), []
    for arch, _ in visited:
        val = gt.get(arch)
        if val is not None and val > best:
            best = val
        curve.append(best)
    return curve


def kendall(xs, ys):
    from scipy.stats import kendalltau
    tau, _ = kendalltau(xs, ys)
    return float(tau)


def op_profile(arch_strs):
    counts = {op: 0 for op in OPS}
    for a in arch_strs:
        for op in OP_RE.findall(a):
            if op in counts:
                counts[op] += 1
    total = sum(counts.values()) or 1
    return {op: round(c / total, 4) for op, c in counts.items()}


def method_report(run, gt, top_k=5):
    uniq = dedupe_keep_first(run['visited'])
    gt_vals = [(a, gt[a]) for a, _ in uniq if a in gt]
    ranked = sorted(gt_vals, key=lambda t: t[1], reverse=True)
    top = ranked[:top_k]
    scored = [(s, gt[a]) for a, s in uniq
              if s is not None and a in gt]
    front_gt = [gt[a] for a in run['front'] if a in gt]
    # Best archs BY THE METHOD'S OWN SCORE (what the search would deploy).
    by_score = sorted((t for t in uniq if t[1] is not None),
                      key=lambda t: t[1], reverse=True)
    picked = [(a, gt.get(a)) for a, _ in by_score[:top_k]]
    return {
        'method': run['method'],
        'n_evals': len(run['visited']),
        'n_unique': len(uniq),
        'best_gt_visited': max((v for _, v in gt_vals), default=None),
        'top{}_mean_gt'.format(top_k):
            round(sum(v for _, v in top) / len(top), 4) if top else None,
        'best_gt_front': max(front_gt, default=None),
        'picked_topk': [{'arch': a, 'gt': v} for a, v in picked],
        'best_gt_picked': max((v for _, v in picked if v is not None),
                              default=None),
        'fidelity_kt': round(kendall([s for s, _ in scored],
                                     [v for _, v in scored]), 4)
        if len(scored) >= 3 else None,
        'top_archs': [a for a, _ in top],
        'visited_set': [a for a, _ in uniq],
        'op_profile_topk': op_profile([a for a, _ in top]),
        'convergence': convergence_curve(run['visited'], gt),
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 4) if a | b else 0.0


def random_baseline(gt, budget, draws=1000, seed=0):
    rng = random.Random(seed)
    keys = list(gt)
    maxes = []
    for _ in range(draws):
        sample = rng.sample(keys, budget)
        maxes.append(max(gt[a] for a in sample))
    maxes.sort()
    n = len(maxes)
    return {'mean_best': round(sum(maxes) / n, 4),
            'p05': round(maxes[int(0.05 * n)], 4),
            'p95': round(maxes[int(0.95 * n)], 4),
            'budget': budget, 'draws': draws}


def make_plots(reports, baseline, gt_optimum, out_prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in reports:
        ax.plot(range(1, len(r['convergence']) + 1), r['convergence'],
                label=r['method'])
    ax.axhline(gt_optimum, color='k', ls=':', lw=1, label='GT optimum')
    ax.axhline(baseline['mean_best'], color='gray', ls='--', lw=1,
               label=f"random x{baseline['budget']}")
    ax.set_xlabel('evaluation'); ax.set_ylabel('best NB201 acc so far')
    ax.set_title('Guided-search convergence (seed 42, cifar10)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_prefix + '_convergence.png', dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r['method'] for r in reports]
    vals = [r['best_gt_visited'] or 0 for r in reports]
    ax.bar(names, vals)
    ax.axhline(gt_optimum, color='k', ls=':', lw=1)
    ax.axhspan(baseline['p05'], baseline['p95'], color='gray', alpha=0.25,
               label='random-search 5-95%')
    lo = min(baseline['p05'], min(vals)) - 0.5
    ax.set_ylim(lo, gt_optimum + 0.3)
    ax.set_ylabel('best NB201 acc found'); ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_prefix + '_best.png', dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n = len(reports)
    mat = [[jaccard(reports[i]['visited_set'], reports[j]['visited_set'])
            for j in range(n)] for i in range(n)]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{mat[i][j]:.2f}', ha='center', va='center',
                    color='w' if mat[i][j] < 0.6 else 'k', fontsize=7)
    ax.set_title('Visited-set Jaccard overlap')
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_prefix + '_overlap.png', dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True,
                   help='parent dir containing the guided run dirs (search-*)')
    p.add_argument('--gt', default='misc/nb201_cifar10_gt.json')
    p.add_argument('--gt-key', default='cifar10_test')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--out-prefix', default='')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)

    gt = load_gt(args.gt, args.gt_key)
    gt_optimum = max(gt.values())
    run_dirs = sorted(glob.glob(os.path.join(args.runs, 'search-*')))
    run_dirs = [d for d in run_dirs
                if os.path.exists(os.path.join(d, 'summary.json'))]
    if not run_dirs:
        print(f'error: no search-*/summary.json under {args.runs}')
        return 1

    reports = [method_report(load_run(d), gt, args.top_k) for d in run_dirs]
    budget = max(r['n_evals'] for r in reports)
    baseline = random_baseline(gt, budget)

    out_prefix = args.out_prefix or os.path.join(args.runs, 'comparison')
    n = len(reports)
    overlap = {f"{reports[i]['method']}|{reports[j]['method']}":
               {'visited': jaccard(reports[i]['visited_set'],
                                   reports[j]['visited_set']),
                'topk': jaccard(reports[i]['top_archs'],
                                reports[j]['top_archs'])}
               for i in range(n) for j in range(i + 1, n)}
    payload = {'gt_key': args.gt_key, 'gt_optimum': gt_optimum,
               'random_baseline': baseline, 'methods': reports,
               'overlap': overlap}
    with open(out_prefix + '.json', 'w') as f:
        json.dump(payload, f, indent=2)

    fields = ['method', 'n_evals', 'n_unique', 'best_gt_visited',
              f'top{args.top_k}_mean_gt', 'best_gt_front', 'best_gt_picked',
              'fidelity_kt']
    with open(out_prefix + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(reports, key=lambda r: -(r['best_gt_visited'] or 0)):
            w.writerow(r)

    if not args.no_plots:
        make_plots(reports, baseline, gt_optimum, out_prefix)

    print(f"GT optimum {gt_optimum:.2f} | random x{budget}: "
          f"mean best {baseline['mean_best']:.2f} "
          f"[{baseline['p05']:.2f}, {baseline['p95']:.2f}]")
    print(f"{'method':>11} {'best':>6} {'top-mean':>8} {'picked':>6} "
          f"{'front':>6} {'fid-KT':>7} {'uniq':>5}")
    for r in sorted(reports, key=lambda r: -(r['best_gt_visited'] or 0)):
        print(f"{r['method']:>11} {r['best_gt_visited'] or 0:>6.2f} "
              f"{r[f'top{args.top_k}_mean_gt'] or 0:>8.2f} "
              f"{r['best_gt_picked'] or 0:>6.2f} "
              f"{r['best_gt_front'] or 0:>6.2f} "
              f"{r['fidelity_kt'] if r['fidelity_kt'] is not None else 0:>7.3f} "
              f"{r['n_unique']:>5}")
    print(f'wrote {out_prefix}.json / .csv'
          + ('' if args.no_plots else ' / _convergence.png / _best.png / _overlap.png'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
