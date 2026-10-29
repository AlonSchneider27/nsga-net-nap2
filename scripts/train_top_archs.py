#!/usr/bin/env python3
"""Fully train the top architectures found by a (guided) search run.

Takes a finished run's summary.json (or final_pop.json), picks the top-k
architectures by the guiding method's score — or the final Pareto set — and
trains each with the NB201 20-epoch recipe, reporting true accuracies for
cross-method comparison. NB201-only (arch reconstruction needs arch_str).

Usage from project root::

    python scripts/train_top_archs.py --summary experiments/method_guided/c10/search-... \
        --top-k 5 --dataset cifar10 --epochs 20

    python scripts/train_top_archs.py --final-pop <run>/final_pop.json --from-final-pop \
        --dataset cifar10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARCH_STR_RE = re.compile(r"arch_str=['\"]([^'\"]+)['\"]")


def extract_arch_str(genotype_repr):
    """arch_str out of an NB201Genotype repr; micro/macro genotypes raise."""
    m = ARCH_STR_RE.search(genotype_repr or '')
    if not m:
        raise ValueError(
            'NB201-only: cannot reconstruct a model from genotype '
            f'{genotype_repr!r} (no arch_str=...)')
    return m.group(1)


def load_summary(path):
    """Accept a run dir (-> <dir>/summary.json) or a summary.json path."""
    p = str(path)
    if os.path.isdir(p):
        p = os.path.join(p, 'summary.json')
    with open(p) as f:
        return json.load(f)


def resolve_rank_key(architectures, name):
    """Map a method name to ('attr', field) or ('fitness', key).

    Bare method names resolve to the exact fitness key if present, else the
    max-@budget variant (e.g. sotl_e -> sotl_e@23). 'nap2' prefers its
    fitness keys but falls back to the pred_acc attribute. valid_acc and
    pred_acc are addressable directly.
    """
    if name in ('valid_acc', 'pred_acc'):
        return ('attr', name)
    keys = set()
    for arch in architectures.values():
        keys.update((arch.get('fitness') or {}).keys())
    if name in keys:
        return ('fitness', name)
    budgets = []
    for k in keys:
        base, _, suffix = k.rpartition('@')
        if base == name and suffix.isdigit():
            budgets.append(int(suffix))
    if budgets:
        return ('fitness', f'{name}@{max(budgets)}')
    if name == 'nap2':
        return ('attr', 'pred_acc')
    raise KeyError(
        f'rank key {name!r} not found in the summary (available fitness keys: '
        f'{sorted(keys)}); pass --rank-by explicitly')


def _score_of(arch, kind, key):
    if kind == 'attr':
        return arch.get(key)
    return (arch.get('fitness') or {}).get(key)


def select_archs(summary, top_k, rank_by='auto'):
    """Top-k unique archs ranked (desc) by the resolved score key."""
    architectures = summary['architectures']
    name = rank_by
    if rank_by == 'auto':
        name = summary.get('objective_method') or 'pred_acc'
    kind, key = resolve_rank_key(architectures, name)

    scored = []
    for arch_id, arch in architectures.items():
        score = _score_of(arch, kind, key)
        if score is None:
            continue
        scored.append((arch_id, arch, float(score)))
    if not scored:
        raise SystemExit(
            f'error: no architecture carries a {key!r} score — refusing to '
            'rank (pass --rank-by to choose another key; never ranking by '
            'untrained valid_acc silently)')

    scored.sort(key=lambda t: t[2], reverse=True)
    rows, seen = [], set()
    for arch_id, arch, score in scored:
        arch_str = extract_arch_str(arch.get('genotype'))
        if arch_str in seen:
            continue
        seen.add(arch_str)
        rows.append({'rank': len(rows) + 1, 'arch_id': arch_id,
                     'arch_str': arch_str, 'rank_key': key, 'score': score,
                     'valid_acc': arch.get('valid_acc')})
        if top_k and len(rows) >= top_k:
            break
    return rows


def select_from_final_pop(final_pop, top_k=None):
    """Pareto-front entries of final_pop.json, deduped; optional top-k by F[0].

    top_k None/0 = the WHOLE front (the default — truncating it would bias
    the selection toward the guided-score end and drop the low-flops end).
    """
    entries = [e for e in final_pop['population'] if e.get('on_pareto_front')]
    entries.sort(key=lambda e: e['F'][0])
    rows, seen = [], set()
    for e in entries:
        arch_str = e.get('arch_str') or extract_arch_str(e.get('genotype'))
        if arch_str in seen:
            continue
        seen.add(arch_str)
        rows.append({'rank': len(rows) + 1, 'arch_id': None,
                     'arch_str': arch_str, 'rank_key': 'pareto_front(F0)',
                     'score': -e['F'][0], 'valid_acc': None})
    if top_k and len(rows) > top_k:
        print(f'note: truncating Pareto front from {len(rows)} to top-{top_k} '
              'by guided score (dropping the low-flops end)')
        rows = rows[:top_k]
    return rows


def train_selected(rows, args):
    """Train each row's arch with the NB201 recipe; adds trained_acc, t_min."""
    import torch
    from nap2_local_benchmark import train_arch, BATCH_SIZE
    from nap2.search_spaces.nb201_ops import build_nb201_model
    from misc.dataset_configs import (get_config, get_loader_class,
                                      build_search_transforms)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch.backends, 'mps', None) is not None \
                and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    cfg = get_config(args.dataset)
    root = args.data or cfg['data_dir']
    DatasetCls = get_loader_class(args.dataset)
    train_data = DatasetCls(root=root, train=True, download=True,
                            transform=build_search_transforms(cfg, train=True))
    test_data = DatasetCls(root=root, train=False, download=True,
                           transform=build_search_transforms(cfg, train=False))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f'training {len(rows)} archs: dataset={args.dataset} '
          f'epochs={args.epochs} device={device}')
    for row in rows:
        t0 = time.time()
        torch.manual_seed(args.seed)
        model = build_nb201_model(row['arch_str'],
                                  num_classes=cfg['num_classes'], C=16, N=5)
        row['trained_acc'] = train_arch(model, train_loader, test_loader,
                                        args.epochs, device)
        row['t_min'] = round((time.time() - t0) / 60, 1)
        print(f"  #{row['rank']} score={row['score']:.4f} "
              f"trained_acc={row['trained_acc']:.4f} ({row['t_min']}min) "
              f"{row['arch_str']}", flush=True)
    return rows


FIELDS = ['rank', 'arch_id', 'arch_str', 'rank_key', 'score', 'valid_acc',
          'trained_acc', 't_min']


def write_outputs(rows, out_prefix):
    csv_path, json_path = out_prefix + '.csv', out_prefix + '.json'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary', default='',
                   help='run dir or summary.json of the finished search')
    p.add_argument('--final-pop', default='',
                   help='final_pop.json path (with --from-final-pop)')
    p.add_argument('--from-final-pop', action='store_true',
                   help='select the Pareto-front archs from --final-pop '
                        'instead of top-k from --summary')
    p.add_argument('--top-k', type=int, default=None,
                   help='how many archs to train. Default: 5 when ranking a '
                        'summary; the WHOLE Pareto front with --from-final-pop')
    p.add_argument('--rank-by', default='auto',
                   help="method name or fitness key; 'auto' = the summary's "
                        "objective_method, falling back to pred_acc")
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'ImageNet16-120'])
    p.add_argument('--data', default='', help='dataset root override')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='',
                   help='output prefix (default: <run dir>/top_archs)')
    args = p.parse_args(argv)

    if args.from_final_pop:
        if not args.final_pop:
            p.error('--from-final-pop needs --final-pop')
        with open(args.final_pop) as f:
            rows = select_from_final_pop(json.load(f), args.top_k)
        base_dir = os.path.dirname(os.path.abspath(args.final_pop))
    else:
        if not args.summary:
            p.error('need --summary (or --final-pop with --from-final-pop)')
        rows = select_archs(load_summary(args.summary),
                            args.top_k if args.top_k is not None else 5,
                            args.rank_by)
        base_dir = args.summary if os.path.isdir(args.summary) \
            else os.path.dirname(os.path.abspath(args.summary))

    print(f"selected {len(rows)} archs by {rows[0]['rank_key']}")
    rows = train_selected(rows, args)

    out_prefix = args.out or os.path.join(base_dir, 'top_archs')
    csv_path, json_path = write_outputs(rows, out_prefix)
    print(f'wrote {csv_path} and {json_path}')

    best = max(rows, key=lambda r: r['trained_acc'])
    print(f"best found: trained_acc={best['trained_acc']:.4f} "
          f"(rank #{best['rank']}) {best['arch_str']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
