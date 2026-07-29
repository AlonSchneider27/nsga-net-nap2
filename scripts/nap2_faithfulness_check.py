#!/usr/bin/env python3
"""Measure integration faithfulness: our NAP2Predictor vs michael's lookup tables.

Scores K architectures straight from the NB201 catalog with the in-repo
NAP2 pipeline (same checkpoints the search uses) and compares per-arch
against the pre-computed lookup predictions (snap5 / snap1). No EA, no
proxy training in the loop -- this isolates "does our pipeline reproduce
michael's pipeline" from everything else.

With --ablate it predicts each arch four ways (same embeddings, four cheap
predict() calls) to attribute the error to each known bug:

                     pad->max_steps    no pad
    normalize=log       (fixed)      isolates padding
    normalize=none   isolates norm   (the old, doubly-broken path)

Interpretation: KT(ours, snap5) near the lookups' own snap1-vs-snap5
self-agreement (~0.88) means the integration is faithful. On the default
--first-n set (the catalog's leading archs are mostly disconnected -> lookup
~0.10, with a few skip-path archs -> ~0.84), the *group separation* is a
sharper signal than KT: a working pipeline must score the dead group near
0.10, not near the predictor's ~0.9 prior.

Example:

    python scripts/nap2_faithfulness_check.py \
        --lookup-snap5 nap2/nap2_log_snap5_cifar10.json \
        --lookup-snap1 nap2/nap2_log_snap1_cifar10.json \
        --first-n 30 --steps 5 --max-steps 31 --ablate \
        --ae-weights-pt     nap2/trained_models/cifar10/ae/weights/best_ae_model.pt \
        --ae-weights-json   nap2/trained_models/cifar10/ae/weights/model_hyper_params.json \
        --ae-gradients-pt   nap2/trained_models/cifar10/ae/gradients/best_ae_model.pt \
        --ae-gradients-json nap2/trained_models/cifar10/ae/gradients/model_hyper_params.json \
        --predictor-pt      nap2/trained_models/cifar10/biGRU/cp/model_state_cp/model.pt \
        --predictor-json    nap2/trained_models/cifar10/biGRU/model_hyper_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.lstm_predictor import LSTMPredictor
from nap2.predictor import NAP2Predictor, resolve_normalize
from nap2.search_spaces.nb201_ops import build_nb201_model
from nap2.training.train_snapshots_nb201 import load_dataset


def load_json_lenient(path: str) -> dict:
    """Parse a json that may carry stray bytes before the opening brace."""
    raw = open(path).read()
    return json.loads(raw[raw.index("{"):])


def load_predictor(a: argparse.Namespace, normalize_override=None) -> NAP2Predictor:
    ae_w = FeatureMapAutoEncoder.load(model_path=a.ae_weights_pt, params_path=a.ae_weights_json)
    ae_g = FeatureMapAutoEncoder.load(model_path=a.ae_gradients_pt, params_path=a.ae_gradients_json)
    pred_params = load_json_lenient(a.predictor_json)
    ae_params = load_json_lenient(a.ae_weights_json) if a.ae_weights_json else None
    if pred_params.get("predictor_type", "lstm") == "bigru":
        net = BiGRUDualPredictor.load(model_path=a.predictor_pt, params_path=a.predictor_json)
    else:
        net = LSTMPredictor.load(model_path=a.predictor_pt, params_path=a.predictor_json)
    if normalize_override is not None:
        normalize, source = normalize_override, "override"
    else:
        normalize, source = resolve_normalize(ae_params, pred_params)
    print(f"predictor loaded: type={pred_params.get('predictor_type')} "
          f"normalize={normalize} (from {source})")
    return NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=net, normalize=normalize)


def count_dead_layers(model, dataloader, device) -> int:
    """One fwd/bwd batch on a fresh copy; count Conv2d/Linear params whose
    gradient is missing or all-zero (the rows that go zero in the map grid)."""
    import copy
    m = copy.deepcopy(model).to(device)
    m.train()
    inputs, targets = next(iter(dataloader))
    outputs = m(inputs.to(device))
    if isinstance(outputs, tuple):
        outputs = outputs[-1]
    nn.CrossEntropyLoss()(outputs, targets.to(device)).backward()
    dead = 0
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            g = mod.weight.grad
            if g is None or bool(torch.all(g == 0)):
                dead += 1
    return dead


def pad_to(seq: torch.Tensor, n: int) -> torch.Tensor:
    if n and n > seq.shape[0]:
        pad = torch.zeros(n - seq.shape[0], seq.shape[1], dtype=seq.dtype)
        return torch.cat([seq, pad], dim=0)
    return seq


def summarize(rows, key, label, snap5_l, snap1_l):
    from scipy.stats import kendalltau
    vals = [r[key] for r in rows]
    kt5, _ = kendalltau(vals, snap5_l)
    kt1, _ = kendalltau(vals, snap1_l)
    mae = sum(abs(v - s) for v, s in zip(vals, snap5_l)) / len(vals)
    dead = [v for v, s in zip(vals, snap5_l) if s < 0.5]
    alive = [v for v, s in zip(vals, snap5_l) if s >= 0.5]
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    line = (f"  {label:<22} KT5={kt5:+.3f} KT1={kt1:+.3f} MAE={mae:.3f}  "
            f"range=[{min(vals):.3f},{max(vals):.3f}] sd={sd:.4f}")
    if dead and alive:
        gap = statistics.mean(alive) - statistics.mean(dead)
        line += (f"  | dead={statistics.mean(dead):.3f} alive={statistics.mean(alive):.3f} "
                 f"gap={gap:+.3f}")
    print(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookup-snap5", required=True)
    p.add_argument("--lookup-snap1", default=None)
    p.add_argument("--dataset-dir", default="data")
    p.add_argument("--ae-weights-pt", required=True)
    p.add_argument("--ae-weights-json", required=True)
    p.add_argument("--ae-gradients-pt", required=True)
    p.add_argument("--ae-gradients-json", required=True)
    p.add_argument("--predictor-pt", required=True)
    p.add_argument("--predictor-json", required=True)
    p.add_argument("--first-n", type=int, default=None,
                   help="take the first N archs in lookup order (diagnostic set); "
                        "default is a stratified sample of --n-archs")
    p.add_argument("--n-archs", type=int, default=20)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=31)
    p.add_argument("--ablate", action="store_true",
                   help="also predict with normalize=none and without padding")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default=None)
    a = p.parse_args()

    if a.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = a.device

    snap5 = load_json_lenient(a.lookup_snap5)
    snap1 = load_json_lenient(a.lookup_snap1) if a.lookup_snap1 else {}

    if a.first_n:
        picks = list(snap5.keys())[:a.first_n]
    else:
        rng = random.Random(a.seed)
        by_value = sorted(snap5, key=snap5.get)
        stride = len(by_value) / a.n_archs
        picks = [by_value[min(int(i * stride) + rng.randrange(max(1, int(stride))),
                              len(by_value) - 1)] for i in range(a.n_archs)]

    predictor = load_predictor(a)
    predictor_raw = load_predictor(a, normalize_override="none") if a.ablate else None
    loader = load_dataset(a.dataset_dir, dataset_name="cifar10", batch_size=256)

    print(f"\nn={len(picks)} steps={a.steps} max_steps={a.max_steps} device={device}")
    cols = f"{'log+pad':>8} {'log+nopad':>10} {'raw+pad':>8} {'raw+nopad':>10}" if a.ablate else f"{'ours':>8}"
    print(f"{'#':>3} {cols} {'snap5':>7} {'snap1':>7} {'dead':>4}  arch_str")

    rows, s5_l, s1_l = [], [], []
    for i, arch in enumerate(picks):
        torch.manual_seed(0)
        model = build_nb201_model(arch, num_classes=10, C=16, N=5)
        dead = count_dead_layers(model, loader, device)

        emb = predictor.get_embeddings(model.to(device), loader, steps=a.steps)
        r = {"arch": arch, "log+pad": float(predictor._lstm.predict(pad_to(emb, a.max_steps)))}
        if a.ablate:
            r["log+nopad"] = float(predictor._lstm.predict(emb))
            torch.manual_seed(0)
            m2 = build_nb201_model(arch, num_classes=10, C=16, N=5)
            emb_raw = predictor_raw.get_embeddings(m2.to(device), loader, steps=a.steps)
            r["raw+pad"] = float(predictor_raw._lstm.predict(pad_to(emb_raw, a.max_steps)))
            r["raw+nopad"] = float(predictor_raw._lstm.predict(emb_raw))
        r["dead"] = dead
        rows.append(r)
        s5_l.append(float(snap5[arch]))
        s1_l.append(float(snap1.get(arch, float("nan"))))

        if a.ablate:
            vals = (f"{r['log+pad']:8.4f} {r['log+nopad']:10.4f} "
                    f"{r['raw+pad']:8.4f} {r['raw+nopad']:10.4f}")
        else:
            vals = f"{r['log+pad']:8.4f}"
        print(f"{i:>3} {vals} {s5_l[-1]:7.4f} {s1_l[-1]:7.4f} {dead:>4}  {arch}", flush=True)

    print(f"\n--- summary (n={len(rows)}) ---")
    variants = ["log+pad", "log+nopad", "raw+pad", "raw+nopad"] if a.ablate else ["log+pad"]
    for v in variants:
        summarize(rows, v, v, s5_l, s1_l)
    print("  reference: catalog-wide KT(snap1,snap5) ~ +0.88 (the ceiling)")
    if any(s < 0.5 for s in s5_l) and any(s >= 0.5 for s in s5_l):
        d = [s for s in s5_l if s < 0.5]
        al = [s for s in s5_l if s >= 0.5]
        print(f"  lookup truth: dead={statistics.mean(d):.3f} (n={len(d)})  "
              f"alive={statistics.mean(al):.3f} (n={len(al)})  gap={statistics.mean(al)-statistics.mean(d):+.3f}")

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arch_str"] + variants + ["dead", "snap5", "snap1"])
            for r, s5v, s1v in zip(rows, s5_l, s1_l):
                w.writerow([r["arch"]] + [r[v] for v in variants] + [r["dead"], s5v, s1v])
        print(f"csv -> {a.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
