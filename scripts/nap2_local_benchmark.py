#!/usr/bin/env python3
"""Local nap2 sanity benchmark: predict -> train -> Kendall tau.

Self-contained check that nap2 works, with no EA and no lookup tables:
sample N random NB201 architectures, get nap2 predictions for each at
several step counts (5/10/15 by default), then actually train the same
architectures with the search's NB201 recipe (SGD lr 0.1 / momentum 0.9 /
nesterov / wd 5e-4, batch 256, cosine T_max=200, grad clip 5) and report
KT(prediction, trained accuracy) per step count.

Embeddings are collected once per arch at max(steps) snapshots; smaller
step counts are strict prefixes (snapshots land at batches 100, 200, ...),
each zero-padded to --max-steps before prediction (predict_anytime protocol).

Example (checkpoints under gitignored trained_models/):

    .venv/bin/python scripts/nap2_local_benchmark.py \
        --n-archs 30 --steps-list 5,10,15 --epochs 20 \
        --ae-weights-pt   trained_models/nap2_cifar10/ae_weights/best_ae_model.pt \
        --ae-weights-json trained_models/nap2_cifar10/ae_weights/model_hyper_params.json \
        --ae-gradients-pt   trained_models/nap2_cifar10/ae_gradients/best_ae_model.pt \
        --ae-gradients-json trained_models/nap2_cifar10/ae_gradients/model_hyper_params.json \
        --predictor-pt   trained_models/nap2_cifar10/predictor/model.pt \
        --predictor-json trained_models/nap2_cifar10/predictor/model_hyper_params.json
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from nap2.search_spaces.nb201_ops import build_nb201_model
from nap2.training.train_snapshots_nb201 import load_dataset
from search import nb201_encoding

from nap2_faithfulness_check import load_predictor  # same flag/attr names


# --- NB201 recipe (train_search.py parity for search_space == 'nb201') ----
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NESTEROV = True
BATCH_SIZE = 256
T_MAX = 200
GRAD_CLIP = 5


def sample_arch_strs(n: int, seed: int) -> list:
    """Random unique NB201 arch_strs via the search's own encoding."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        genome = [rng.randrange(len(nb201_encoding.NB201_PRIMITIVES))
                  for _ in range(nb201_encoding.N_EDGES)]
        arch = nb201_encoding.decode(genome).arch_str
        if arch not in seen:
            seen.add(arch)
            out.append(arch)
    return out


def predict_prefixes(predictor, model, loader, steps_list, max_steps):
    """One embedding collection at max(steps_list); predict each prefix."""
    emb = predictor.get_embeddings(model, loader, steps=max(steps_list))
    preds = {}
    for s in steps_list:
        seq = emb[:s]
        if max_steps and max_steps > seq.shape[0]:
            pad = torch.zeros(max_steps - seq.shape[0], seq.shape[1], dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        preds[s] = float(predictor._lstm.predict(seq))
    return preds


def train_arch(model, train_loader, test_loader, epochs, device) -> float:
    """Train with the NB201 recipe; return test-set top-1 accuracy in [0, 1]."""
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=NESTEROV)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_MAX)

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            correct += int((outputs.argmax(dim=1) == targets).sum())
            total += targets.numel()
    return correct / total


def report(rows, steps_list):
    """KT/Spearman per step count + prediction-collapse stats."""
    from scipy.stats import kendalltau, spearmanr
    valid = [r["valid_acc"] for r in rows]
    print(f"\n--- summary over n={len(rows)} archs "
          f"(valid_acc range [{min(valid):.4f}, {max(valid):.4f}]) ---")
    for s in steps_list:
        preds = [r["pred"][s] for r in rows]
        kt = sr = float("nan")
        if len(rows) >= 3:
            kt, _ = kendalltau(preds, valid)
            sr, _ = spearmanr(preds, valid)
        sd = statistics.pstdev(preds) if len(preds) > 1 else 0.0
        print(f"steps={s:>2}: KT={kt:+.4f}  Spearman={sr:+.4f}  "
              f"pred range [{min(preds):.4f}, {max(preds):.4f}] sd={sd:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-archs", type=int, default=30)
    p.add_argument("--steps-list", default="5,10,15")
    p.add_argument("--max-steps", type=int, default=31,
                   help="zero-pad each prefix to this length before prediction")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-dir", default="data")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"),
                   help="training device")
    p.add_argument("--score-device", default="same", choices=("same", "cpu"),
                   help="device for nap2's partial-training during scoring. 'same' "
                        "(default) uses the training device — the float64 AE/GRU "
                        "stages run on CPU numpy regardless, only the model's "
                        "forward/backward runs on-device (25x faster than CPU on "
                        "MPS). Use 'cpu' if on-device scoring misbehaves.")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--ae-weights-pt", required=True)
    p.add_argument("--ae-weights-json", required=True)
    p.add_argument("--ae-gradients-pt", required=True)
    p.add_argument("--ae-gradients-json", required=True)
    p.add_argument("--predictor-pt", required=True)
    p.add_argument("--predictor-json", required=True)
    a = p.parse_args()

    steps_list = sorted(int(s) for s in a.steps_list.split(","))
    if a.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = a.device

    archs = sample_arch_strs(a.n_archs, a.seed)
    predictor = load_predictor(a)
    train_loader = load_dataset(a.dataset_dir, dataset_name="cifar10",
                                batch_size=BATCH_SIZE)
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_data = torchvision.datasets.CIFAR10(root=a.dataset_dir, train=False,
                                             download=True, transform=test_tf)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    print(f"n_archs={a.n_archs} steps={steps_list} max_steps={a.max_steps} "
          f"epochs={a.epochs} train_device={device} seed={a.seed}")
    header = "  ".join(f"pred@{s:<2}" for s in steps_list)
    print(f"{'#':>3} {header}  {'valid':>7} {'t(min)':>7}  arch_str")

    rows = []
    for i, arch in enumerate(archs):
        t0 = time.time()
        # Search parity: same init seed for every arch (train_search seeds 0).
        torch.manual_seed(0)
        model = build_nb201_model(arch, num_classes=10, C=16, N=5)
        train_copy = copy.deepcopy(model)  # pristine init for training

        score_device = device if a.score_device == "same" else "cpu"
        preds = predict_prefixes(predictor, model.to(score_device), train_loader,
                                 steps_list, a.max_steps)
        valid = train_arch(train_copy, train_loader, test_loader,
                           a.epochs, device)

        rows.append({"arch": arch, "pred": preds, "valid_acc": valid})
        pred_s = "  ".join(f"{preds[s]:7.4f}" for s in steps_list)
        print(f"{i:>3} {pred_s}  {valid:7.4f} {(time.time() - t0) / 60:7.1f}  {arch}",
              flush=True)
        if len(rows) >= 5 and len(rows) % 5 == 0:
            report(rows, steps_list)

    report(rows, steps_list)

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arch_str"] + [f"pred_{s}" for s in steps_list] + ["valid_acc"])
            for r in rows:
                w.writerow([r["arch"]] + [r["pred"][s] for s in steps_list]
                           + [r["valid_acc"]])
        print(f"csv written to {a.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
