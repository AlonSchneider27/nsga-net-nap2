#!/usr/bin/env python3
"""Scrape per-architecture metrics from an NSGA-Net + nap2 search log.

Reads a search run's ``log.txt`` and writes a JSON summary mapping each
architecture id to its valid_acc, pred_acc, flops, param_size_mb, and
genotype (verbatim ``Genotype(...)`` repr).

Usage from project root::

    # input is the run directory (resolves to <run_dir>/log.txt)
    python scripts/summarize_search.py experiments/cifar100/search-...-20260508-...

    # or input is the log file directly
    python scripts/summarize_search.py path/to/log.txt -o my_summary.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from anywhere; project root holds the misc package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misc.log_summary import resolve_log_path, write_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to either the run directory or its log.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input>/summary.json when input "
             "is a directory, or summary.json next to the log file otherwise.",
    )
    args = parser.parse_args()

    try:
        log_path = resolve_log_path(args.input)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output is None:
        # Place summary.json alongside the log file by default.
        output = log_path.parent / "summary.json"
    else:
        output = args.output

    payload = write_summary(log_path, output)
    architectures = payload["architectures"]
    metrics = payload["metrics"]

    n_failed = sum(1 for v in architectures.values() if v["pred_acc"] is None)
    n_missing_flops = sum(1 for v in architectures.values() if v["flops"] is None)
    n_missing_param = sum(1 for v in architectures.values() if v["param_size_mb"] is None)
    n_missing_geno = sum(1 for v in architectures.values() if v["genotype"] is None)

    print(f"Wrote {len(architectures)} architectures to {output}")
    if n_failed:
        print(f"  - {n_failed} arch(s) had pred_acc=n/a (predictor failed)")
    if n_missing_flops:
        print(f"  warning: {n_missing_flops} arch(s) missing flops")
    if n_missing_param:
        print(f"  warning: {n_missing_param} arch(s) missing param_size_mb")
    if n_missing_geno:
        print(f"  warning: {n_missing_geno} arch(s) missing genotype")

    print()
    print("Ranking metrics (predicted vs valid_acc):")
    if "error" in metrics:
        print(f"  error: {metrics['error']}")
    elif metrics.get("num_architectures", 0) < 2:
        print(f"  not enough paired observations to compute correlation "
              f"(num_architectures={metrics.get('num_architectures', 0)})")
    else:
        kt = metrics["kendall_tau"]
        sr = metrics["spearman_rho"]
        top = metrics["top_10pct_accuracy"]
        print(f"  kendall_tau         = {kt:.4f}" if kt is not None else "  kendall_tau         = null")
        print(f"  spearman_rho        = {sr:.4f}" if sr is not None else "  spearman_rho        = null")
        print(f"  top_10pct_accuracy  = {top:.4f}" if top is not None else "  top_10pct_accuracy  = null")
        print(f"  num_architectures   = {metrics['num_architectures']}")
        print(f"  num_failed_predictions = {metrics['num_failed_predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
