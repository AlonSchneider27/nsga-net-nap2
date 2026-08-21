#!/usr/bin/env python3
"""Sanity benchmark for the learning-curve fitness baselines.

Ground truth comes from already-finished search runs: the per-arch 20-epoch
NB201-recipe valid_acc scraped from one or more log/.out files (NOT the
nap2_log_snap*.json lookups — those are nap2's own predictions). A stratified
sample of those archs is partially trained once at max(--steps-list) snapshots
with the shared fitness trace runner, every method is scored on every budget
prefix, and Kendall tau / Spearman vs the scraped valid_acc are printed per
(method, budget).

All taus should be positive (sign bugs are the classic failure mode);
SoTL-E at 100 minibatches on CIFAR-10 landed at tau ~= 0.48 in the NAP2
paper's Table 4 (vs full NB201 accuracy — with 20-epoch ground truth the
number is only a loose anchor).

Example:

    .venv/bin/python scripts/lc_baselines_benchmark.py \
        --gt-logs experiment_results/cifar10/*.out \
        --n-archs 20 --steps-list 1,3,5,10,15 --target-epochs 20 \
        --lcpfn-ckpt trained_models/lcpfn/pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from fitness import build_scorers
from fitness.trace import TrainingTrace, run_partial_train
from misc.log_summary import scrape
from nap2.search_spaces.nb201_ops import build_nb201_model
from nap2.training.evaluate import compute_metrics
from nap2.training.train_snapshots_nb201 import load_dataset

BATCH_SIZE = 256
ARCH_STR_RE = re.compile(r"arch_str=['\"]([^'\"]+)['\"]")


def scrape_ground_truth(log_paths):
    """{arch_str: mean 20-epoch valid_acc in [0,1]} across all given logs."""
    by_arch = {}
    for path in log_paths:
        for entry in scrape(path).values():
            genotype = entry.get("genotype") or ""
            m = ARCH_STR_RE.search(genotype)
            if not m or entry.get("valid_acc") is None:
                continue
            by_arch.setdefault(m.group(1), []).append(entry["valid_acc"])
    return {arch: sum(v) / len(v) for arch, v in by_arch.items()}


def stratified_sample(gt, n, seed):
    """Spread n archs across the valid_acc range (stride + seeded jitter)."""
    rng = np.random.RandomState(seed)
    ordered = sorted(gt, key=gt.get)
    if len(ordered) <= n:
        return ordered
    stride = len(ordered) / n
    picks = []
    for i in range(n):
        j = int(i * stride + rng.uniform(0, stride))
        picks.append(ordered[min(j, len(ordered) - 1)])
    return list(dict.fromkeys(picks))


def prefix_trace(trace, steps):
    """Truncate a max-budget trace to a smaller snapshot budget."""
    mb = steps * trace.snapshot_interval
    curve = trace.val_acc_curve[:steps]
    if curve:
        final = curve[-1]
    elif mb == len(trace.minibatch_losses):
        # Curve-free mode (no lce_m/lc_pfn selected): the runner still
        # measured final_val_acc at budget end, valid for the full budget.
        final = trace.final_val_acc
    else:
        # Smaller budgets have no val measurement in curve-free mode.
        final = None
    return TrainingTrace(
        minibatch_losses=trace.minibatch_losses[:mb],
        val_acc_curve=curve,
        final_val_acc=final,
        epoch_len=trace.epoch_len,
        snapshot_interval=trace.snapshot_interval,
        times=trace.times,
    )


def report(rows, gt, methods, steps_list):
    import math
    print(f"\n--- KT/Spearman vs scraped valid_acc over n={len(rows)} archs ---")
    print(f"{'budget':>8} " + " ".join(f"{m:>12}" for m in methods))
    any_nonpositive = False
    for s in steps_list:
        cells = []
        for method in methods:
            preds = {r['arch']: r['scores'][s].get(method) for r in rows
                     if r['scores'][s].get(method) is not None}
            truth = {a: gt[a] for a in preds}
            if len(preds) >= 3:
                kt = compute_metrics(preds, truth)['kendall_tau']
                if kt is not None and math.isfinite(kt):
                    if kt <= 0:
                        any_nonpositive = True
                    cells.append(f"{kt:+.3f}")
                else:
                    # NaN tau (constant scores etc.) is as suspicious as a
                    # negative one — never let it pass as an all-clear.
                    any_nonpositive = True
                    cells.append("   nan")
            else:
                cells.append("  n/a")
        print(f"{s * 100:>6}mb " + " ".join(f"{c:>12}" for c in cells))
    if any_nonpositive:
        print("\n*** WARNING: non-positive or NaN Kendall tau detected — check "
              "sign conventions / method implementations before trusting runs. ***")
    else:
        print("\nAll taus positive.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-logs", nargs="+", required=True,
                   help="search log/.out files to scrape 20-epoch valid_acc from")
    p.add_argument("--n-archs", type=int, default=50)
    p.add_argument("--steps-list", default="1,3,5,10,15",
                   help="snapshot budgets (x100 minibatches each)")
    p.add_argument("--methods", default="all")
    p.add_argument("--lcpfn-ckpt", default="")
    p.add_argument("--target-epochs", type=int, default=20)
    p.add_argument("--dataset-dir", default="data")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()

    steps_list = sorted(int(s) for s in a.steps_list.split(","))
    if a.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = a.device

    gt = scrape_ground_truth(a.gt_logs)
    if len(gt) < 3:
        print(f"error: only {len(gt)} archs with ground truth scraped from "
              f"{len(a.gt_logs)} logs — need at least 3")
        return 1
    archs = stratified_sample(gt, a.n_archs, a.seed)
    scorers = build_scorers(a.methods, lcpfn_ckpt=a.lcpfn_ckpt,
                            target_epochs=a.target_epochs)
    methods = [s.name for s in scorers]
    init_scorers = [s for s in scorers if getattr(s, 'needs_init_model', False)]
    trace_scorers = [s for s in scorers if not getattr(s, 'needs_init_model', False)]
    need_curve = any(s.needs_val_curve for s in trace_scorers)
    need_final = any(s.needs_final_val for s in trace_scorers)

    train_loader = load_dataset(a.dataset_dir, dataset_name="cifar10",
                                batch_size=BATCH_SIZE)
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_data = torchvision.datasets.CIFAR10(root=a.dataset_dir, train=False,
                                             download=True, transform=test_tf)
    valid_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"gt archs={len(gt)} sampled={len(archs)} budgets={steps_list} "
          f"methods={methods} device={device} seed={a.seed}")

    np.random.seed(a.seed)
    zc_batch = None
    if init_scorers:
        zc_inputs, zc_targets = next(iter(train_loader))
        zc_batch = (zc_inputs.to(device), zc_targets.to(device))

    def checked(name, s, raw):
        """float() + non-finite guard shared by both scorer families.

        A diverged arch yields NaN losses -> NaN sotl/sotl_e (and synflow can
        overflow to inf on double); treat non-finite scores as failures so
        they can't poison the KT columns.
        """
        value = float(raw)
        if value != value or value in (float('inf'), float('-inf')):
            print(f"  {name}@{s}: non-finite score, dropped")
            return None
        return value

    rows = []
    for i, arch in enumerate(archs):
        t0 = time.time()
        torch.manual_seed(0)   # search parity: same init seed per arch
        model = build_nb201_model(arch, num_classes=10, C=16, N=5).to(device)

        # Zero-cost proxies score the untrained net once; their value is
        # budget-independent and fills every budget column below.
        init_values = {}
        for scorer in init_scorers:
            try:
                init_values[scorer.name] = checked(
                    scorer.name, 'init',
                    scorer.score_init(model, zc_batch[0], zc_batch[1]))
            except Exception as e:
                print(f"  {scorer.name}@init: failed ({e})")
                init_values[scorer.name] = None

        trace = None
        if trace_scorers:
            trace = run_partial_train(model, train_loader, valid_loader,
                                      max(steps_list) * 100,
                                      need_val_curve=need_curve,
                                      need_final_val=need_final)
        scores = {}
        for s in steps_list:
            scores[s] = dict(init_values)
            if trace is None:
                continue
            sub = prefix_trace(trace, s)
            for scorer in trace_scorers:
                try:
                    scores[s][scorer.name] = checked(scorer.name, s,
                                                     scorer.score(sub))
                except Exception as e:
                    print(f"  {scorer.name}@{s}: failed ({e})")
                    scores[s][scorer.name] = None
        rows.append({"arch": arch, "scores": scores})
        top = scores[max(steps_list)]
        summary = " ".join(f"{m}={top[m]:.4f}" for m in methods
                           if top.get(m) is not None)
        t_train = trace.times['train'] if trace is not None else 0.0
        t_val = trace.times['val'] if trace is not None else 0.0
        print(f"{i:>3} valid={gt[arch]:.4f} t={(time.time()-t0)/60:.1f}min "
              f"t_train={t_train:.0f}s t_val={t_val:.0f}s "
              f"| @{max(steps_list)}: {summary}", flush=True)
        if len(rows) >= 5 and len(rows) % 5 == 0:
            report(rows, gt, methods, steps_list)

    report(rows, gt, methods, steps_list)

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arch_str", "valid_acc", "budget_steps"] + methods)
            for r in rows:
                for s in steps_list:
                    w.writerow([r["arch"], gt[r["arch"]], s]
                               + [r["scores"][s].get(m) for m in methods])
        print(f"csv written to {a.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
