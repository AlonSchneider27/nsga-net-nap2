"""Evaluate NAP2 predictions against ground truth.

Computes ranking correlation metrics (Kendall Tau, Spearman rho) and
top-K accuracy between predicted and actual architecture performance.
Also provides train/test split generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_metrics(
    predicted: Dict[str, float],
    ground_truth: Dict[str, float],
) -> Dict[str, float]:
    """Compute ranking correlation metrics.

    Args:
        predicted: {arch_id: predicted_score}
        ground_truth: {arch_id: actual_accuracy}

    Returns:
        Dict with kendall_tau, spearman_rho, top_10pct_accuracy.
    """
    # Align by common keys
    common = sorted(set(predicted) & set(ground_truth))
    pred_vals = np.array([predicted[k] for k in common])
    true_vals = np.array([ground_truth[k] for k in common])

    kt, _ = kendalltau(pred_vals, true_vals)
    sr, _ = spearmanr(pred_vals, true_vals)

    # Top-10% accuracy: what fraction of predicted top-10% are actually in true top-10%
    n = len(common)
    top_k = max(1, n // 10)
    pred_top_ids = set(np.array(common)[np.argsort(-pred_vals)[:top_k]])
    true_top_ids = set(np.array(common)[np.argsort(-true_vals)[:top_k]])
    top_10_acc = len(pred_top_ids & true_top_ids) / top_k

    return {
        "kendall_tau": float(kt),
        "spearman_rho": float(sr),
        "top_10pct_accuracy": float(top_10_acc),
        "num_architectures": n,
    }


def generate_split(
    arch_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate deterministic 80/20 train/test split."""
    rng = np.random.RandomState(seed)
    ids = sorted(arch_ids)
    rng.shuffle(ids)
    split = int(len(ids) * train_ratio)
    return {"train": ids[:split], "test": ids[split:]}


def generate_kfold_splits(
    arch_ids: List[str],
    k: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """Generate k-fold splits with train/val/test in each fold.

    Each fold has ~1/k architectures as test. The remaining architectures
    are split into train and val using val_ratio.

    Args:
        arch_ids: All architecture IDs.
        k: Number of folds.
        val_ratio: Fraction of non-test architectures to use for validation.
        seed: Random seed for deterministic shuffling.

    Returns:
        List of k dicts, each with 'train', 'val', 'test' keys.
    """
    rng = np.random.RandomState(seed)
    ids = sorted(arch_ids)
    rng.shuffle(ids)

    fold_size = len(ids) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(ids)
        test_ids = ids[start:end]
        train_val_ids = ids[:start] + ids[end:]

        # Split train_val into train and val (deterministic per fold)
        fold_rng = np.random.RandomState(seed + i)
        train_val_shuffled = list(train_val_ids)
        fold_rng.shuffle(train_val_shuffled)
        val_count = int(len(train_val_shuffled) * val_ratio)
        val_ids = sorted(train_val_shuffled[:val_count])
        train_ids = sorted(train_val_shuffled[val_count:])

        folds.append({
            "fold": i + 1,
            "train": train_ids,
            "val": val_ids,
            "test": sorted(test_ids),
        })

    return folds


def main():
    parser = argparse.ArgumentParser(description="NAP2 evaluation tools")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    eval_parser.add_argument("--predictions-json", type=str, required=True,
                             help="JSON: {arch_id: predicted_score}")
    eval_parser.add_argument("--ground-truth-json", type=str, required=True,
                             help="JSON: {arch_id: actual_accuracy}")
    eval_parser.add_argument("--output-json", type=str, default=None,
                             help="Save metrics to JSON file")

    # Split subcommand
    split_parser = subparsers.add_parser("split", help="Generate train/test split")
    split_parser.add_argument("--results-json", required=True,
                              help="JSON with arch IDs (keys used for split)")
    split_parser.add_argument("--output", required=True,
                              help="Output split JSON file")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--seed", type=int, default=42)

    # K-fold subcommand
    kfold_parser = subparsers.add_parser("kfold", help="Generate k-fold splits")
    kfold_parser.add_argument("--results-json", required=True,
                              help="JSON with arch IDs (keys used for split)")
    kfold_parser.add_argument("--output-dir", required=True,
                              help="Output directory for fold JSON files")
    kfold_parser.add_argument("--k", type=int, default=4,
                              help="Number of folds (default: 4)")
    kfold_parser.add_argument("--val-ratio", type=float, default=0.2,
                              help="Fraction of train set for validation (default: 0.2)")
    kfold_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "evaluate":
        with open(args.predictions_json) as f:
            predicted = json.load(f)
        with open(args.ground_truth_json) as f:
            ground_truth = json.load(f)

        metrics = compute_metrics(predicted, ground_truth)

        print("=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved to {args.output_json}")

    elif args.command == "split":
        with open(args.results_json) as f:
            results = json.load(f)

        split = generate_split(list(results.keys()),
                               train_ratio=args.train_ratio,
                               seed=args.seed)

        with open(args.output, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Split: {len(split['train'])} train, {len(split['test'])} test")
        print(f"Saved to {args.output}")

    elif args.command == "kfold":
        with open(args.results_json) as f:
            results = json.load(f)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        folds = generate_kfold_splits(
            list(results.keys()),
            k=args.k,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        for fold in folds:
            fold_path = output_dir / f"fold{fold['fold']}_split.json"
            with open(str(fold_path), "w") as f:
                json.dump(fold, f, indent=2)
            print(f"Fold {fold['fold']}: {len(fold['train'])} train, "
                  f"{len(fold['val'])} val, {len(fold['test'])} test "
                  f"-> {fold_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
