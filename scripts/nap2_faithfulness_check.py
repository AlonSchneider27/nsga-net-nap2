#!/usr/bin/env python3
"""Measure integration faithfulness: our NAP2Predictor vs michael's lookup tables.

Scores K architectures straight from the NB201 catalog with the in-repo
NAP2 pipeline (same checkpoints the search uses) and compares per-arch
against the pre-computed lookup predictions (snap5 = 5-snapshot variant,
optionally snap1). No EA, no proxy training in the loop — this isolates
"does our pipeline reproduce michael's pipeline" from everything else.

Interpretation: KT(ours, snap5) near the lookup's own snap1-vs-snap5
self-agreement (~0.88) means the integration is faithful; mediocre KT
means a pipeline divergence remains.

Run on the cluster (needs torch + the checkpoint paths):

    python scripts/nap2_faithfulness_check.py \
        --lookup-snap5 /path/to/lookup_tables/seed42/nap2_log_snap5_cifar10.json \
        --lookup-snap1 /path/to/lookup_tables/seed42/nap2_log_snap1_cifar10.json \
        --dataset-dir  data \
        --ae-weights-pt ... --ae-weights-json ... \
        --ae-gradients-pt ... --ae-gradients-json ... \
        --predictor-pt ... --predictor-json ... \
        --n-archs 20 --steps 5 --max-steps 31
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.lstm_predictor import LSTMPredictor
from nap2.predictor import NAP2Predictor
from nap2.search_spaces.nb201_ops import build_nb201_model
from nap2.training.train_snapshots_nb201 import load_dataset


def load_predictor(a: argparse.Namespace) -> NAP2Predictor:
    ae_w = FeatureMapAutoEncoder.load(model_path=a.ae_weights_pt, params_path=a.ae_weights_json)
    ae_g = FeatureMapAutoEncoder.load(model_path=a.ae_gradients_pt, params_path=a.ae_gradients_json)
    with open(a.predictor_json) as f:
        pred_params = json.load(f)
    if pred_params.get("predictor_type", "lstm") == "bigru":
        net = BiGRUDualPredictor.load(model_path=a.predictor_pt, params_path=a.predictor_json)
    else:
        net = LSTMPredictor.load(model_path=a.predictor_pt, params_path=a.predictor_json)
    with open(a.ae_weights_json) as f:
        normalize = json.load(f).get("normalize", "none")
    return NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=net, normalize=normalize)


def count_dead_layers(arch_str: str, dataloader) -> int:
    """One fwd/bwd batch on a fresh model; count Conv2d/Linear params with
    missing or all-zero gradient (the layers whose grid rows go zero)."""
    model = build_nb201_model(arch_str, num_classes=10, C=16, N=5)
    model.train()
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(dataloader))
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[-1]
    criterion(outputs, targets).backward()
    dead = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            g = m.weight.grad
            if g is None or bool(torch.all(g == 0)):
                dead += 1
    return dead


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookup-snap5", required=True, help="snap5 lookup json (arch_str -> pred)")
    p.add_argument("--lookup-snap1", default=None, help="optional snap1 lookup json")
    p.add_argument("--dataset-dir", default="data", help="CIFAR-10 root (downloaded if missing)")
    p.add_argument("--ae-weights-pt", required=True)
    p.add_argument("--ae-weights-json", required=True)
    p.add_argument("--ae-gradients-pt", required=True)
    p.add_argument("--ae-gradients-json", required=True)
    p.add_argument("--predictor-pt", required=True)
    p.add_argument("--predictor-json", required=True)
    p.add_argument("--n-archs", type=int, default=20)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=31,
                   help="zero-pad embeddings to this length (predictor's training max_seq_len)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    snap5 = json.load(open(a.lookup_snap5))
    snap1 = json.load(open(a.lookup_snap1)) if a.lookup_snap1 else {}

    # Stratified sample: sort catalog by snap5 value, take evenly spaced
    # entries (plus jitter) so the sample spans dead archs to top archs.
    rng = random.Random(a.seed)
    by_value = sorted(snap5, key=snap5.get)
    stride = len(by_value) / a.n_archs
    picks = [by_value[min(int(i * stride) + rng.randrange(max(1, int(stride))),
                          len(by_value) - 1)]
             for i in range(a.n_archs)]

    predictor = load_predictor(a)
    dataloader = load_dataset(a.dataset_dir, dataset_name="cifar10", batch_size=256)

    print(f"n={a.n_archs} steps={a.steps} max_steps={a.max_steps} seed={a.seed}")
    print(f"{'#':>3} {'ours':>7} {'snap5':>7} {'snap1':>7} {'|d5|':>6} {'dead':>4}  arch_str")
    ours_l, s5_l, s1_l = [], [], []
    for i, arch in enumerate(picks):
        model = build_nb201_model(arch, num_classes=10, C=16, N=5)
        ours = float(predictor.score(model, dataloader, steps=a.steps,
                                     max_steps=a.max_steps))
        s5 = float(snap5[arch])
        s1v = snap1.get(arch)
        dead = count_dead_layers(arch, dataloader)
        ours_l.append(ours)
        s5_l.append(s5)
        if s1v is not None:
            s1_l.append(float(s1v))
        s1_s = f"{s1v:7.4f}" if s1v is not None else "    n/a"
        print(f"{i:>3} {ours:7.4f} {s5:7.4f} {s1_s} {abs(ours - s5):6.4f} {dead:>4}  {arch}")

    from scipy.stats import kendalltau
    kt5, _ = kendalltau(ours_l, s5_l)
    mae = sum(abs(o - s) for o, s in zip(ours_l, s5_l)) / len(ours_l)
    print(f"\nKT(ours, snap5) = {kt5:+.4f}   MAE = {mae:.4f}")
    if len(s1_l) == len(ours_l):
        kt1, _ = kendalltau(ours_l, s1_l)
        print(f"KT(ours, snap1) = {kt1:+.4f}")
    print("reference ceiling: catalog-wide KT(snap1, snap5) ~ +0.88")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
