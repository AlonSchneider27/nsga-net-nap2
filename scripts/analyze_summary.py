#!/usr/bin/env python3
"""Pretty-print analysis of a finished NSGA-Net + nap2 run summary.

Companion to ``scripts/summarize_search.py``. ``summarize_search`` parses
a search log and writes ``summary.json``; ``analyze_summary`` reads
``summary.json`` and prints metrics, distribution stats, top/bottom-N
architectures, and a small text histogram of ``pred_acc``.

Pure stdlib. Tolerates both the new ``{architectures, metrics}`` envelope
and the older flat ``{arch_id: {...}}`` layout (as produced by
``write_summary`` before the metrics rollup landed).

Usage from project root::

    # input is the run directory (resolves to <run>/summary.json)
    python scripts/analyze_summary.py experiments/cifar100/search-...

    # or the summary.json directly
    python scripts/analyze_summary.py path/to/summary.json --top 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_payload(path: Path) -> Tuple[Dict, Dict]:
    """Return (architectures, metrics) regardless of envelope shape.

    metrics may be ``None`` when the on-disk JSON is the older flat
    layout that predates the metrics rollup.
    """
    if path.is_dir():
        candidate = path / "summary.json"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{path} is a directory but no summary.json found inside it"
            )
        path = candidate

    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "architectures" in raw:
        return raw["architectures"], raw.get("metrics")
    # Old flat layout.
    return raw, None


def _fmt_metric(name: str, value, width: int = 22) -> str:
    if value is None:
        return f"  {name:<{width}} = null"
    if isinstance(value, float):
        return f"  {name:<{width}} = {value:.4f}"
    return f"  {name:<{width}} = {value}"


def _summarize_distribution(label: str, values: List[float], precision: int = 4) -> str:
    if not values:
        return f"{label}: (no data)"
    vmin, vmax = min(values), max(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    fmt = f".{precision}f"
    return (
        f"{label}: "
        f"min={vmin:{fmt}}  "
        f"max={vmax:{fmt}}  "
        f"mean={mean:{fmt}}  "
        f"median={median:{fmt}}  "
        f"std={stdev:{fmt}}  "
        f"n={len(values)}"
    )


def _text_histogram(values: List[float], n_bins: int = 20, width: int = 40) -> str:
    if not values or n_bins < 1:
        return "(no data)"
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return f"  all values = {vmin:.4f} (no spread)"
    bin_w = (vmax - vmin) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - vmin) / bin_w), n_bins - 1)
        counts[idx] += 1
    cmax = max(counts)
    lines = []
    for i, c in enumerate(counts):
        lo = vmin + i * bin_w
        hi = lo + bin_w
        bar = "#" * int(round(c / cmax * width)) if cmax else ""
        lines.append(f"  [{lo:9.4f}, {hi:9.4f}) {c:>5}  {bar}")
    return "\n".join(lines)


def _print_top(architectures: Dict, key: str, top_n: int, label: str,
               filter_none: bool = True) -> None:
    items = list(architectures.items())
    if filter_none:
        items = [(k, v) for k, v in items if v.get(key) is not None]
    if not items:
        print(f"\n{label}: (no architectures with {key})")
        return
    items.sort(key=lambda kv: kv[1][key], reverse=True)
    print(f"\n{label}:")
    print(f"  {'arch':>6}  {'valid_acc':>10}  {'pred_acc':>10}  {'flops':>10}  {'params(MB)':>10}")
    for k, v in items[:top_n]:
        valid = v.get("valid_acc")
        pred = v.get("pred_acc")
        flops = v.get("flops")
        params = v.get("param_size_mb")
        valid_s = f"{valid:.2f}" if valid is not None else "n/a"
        pred_s = f"{pred:.4f}" if pred is not None else "n/a"
        flops_s = f"{flops:.2f}" if flops is not None else "n/a"
        params_s = f"{params:.4f}" if params is not None else "n/a"
        print(f"  {k:>6}  {valid_s:>10}  {pred_s:>10}  {flops_s:>10}  {params_s:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to summary.json or its parent run directory.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Show top-N by valid_acc and by pred_acc (default: 5).",
    )
    parser.add_argument(
        "--bottom",
        type=int,
        default=0,
        help="Also show bottom-N (default: 0, off).",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print a 20-bin text histogram of pred_acc (helps spot bimodality).",
    )
    args = parser.parse_args()

    try:
        architectures, metrics = _load_payload(args.summary)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n_total = len(architectures)
    valid = [v["valid_acc"] for v in architectures.values()
             if v.get("valid_acc") is not None]
    preds = [v["pred_acc"] for v in architectures.values()
             if v.get("pred_acc") is not None]
    flops = [v["flops"] for v in architectures.values()
             if v.get("flops") is not None]
    params = [v["param_size_mb"] for v in architectures.values()
              if v.get("param_size_mb") is not None]
    n_failed = sum(1 for v in architectures.values() if v.get("pred_acc") is None)

    print(f"Architectures parsed       : {n_total}")
    print(f"with valid_acc             : {len(valid)}")
    print(f"with pred_acc (not n/a)    : {len(preds)}")
    print(f"pred_acc=n/a (predictor failed): {n_failed}")

    print()
    print("=== Ranking metrics (predicted vs valid_acc) ===")
    if metrics is None:
        print("  summary.json is the old flat layout; metrics rollup not present.")
        print("  Re-run scripts/summarize_search.py to regenerate with metrics.")
    elif "error" in metrics:
        print(f"  metrics rollup errored at write time: {metrics['error']}")
    else:
        for name in ("kendall_tau", "spearman_rho", "top_10pct_accuracy",
                     "num_architectures", "num_failed_predictions"):
            if name in metrics:
                print(_fmt_metric(name, metrics[name]))

    print()
    print("=== Distributions ===")
    print(_summarize_distribution("valid_acc      ", valid, precision=2))
    print(_summarize_distribution("pred_acc       ", preds, precision=4))
    print(_summarize_distribution("flops          ", flops, precision=2))
    print(_summarize_distribution("param_size_mb  ", params, precision=4))

    if args.histogram:
        print()
        print("=== pred_acc histogram (20 bins) ===")
        print(_text_histogram(preds, n_bins=20, width=40))

    _print_top(architectures, "valid_acc", args.top, f"Top-{args.top} by valid_acc")
    _print_top(architectures, "pred_acc", args.top, f"Top-{args.top} by pred_acc")

    if args.bottom:
        # Show by reversing: pull the head, sort ascending.
        for key, label in (("valid_acc", "valid_acc"), ("pred_acc", "pred_acc")):
            items = [(k, v) for k, v in architectures.items() if v.get(key) is not None]
            items.sort(key=lambda kv: kv[1][key])
            if items:
                print(f"\nBottom-{args.bottom} by {label}:")
                print(f"  {'arch':>6}  {'valid_acc':>10}  {'pred_acc':>10}  {'flops':>10}  {'params(MB)':>10}")
                for k, v in items[:args.bottom]:
                    valid = v.get("valid_acc")
                    pred = v.get("pred_acc")
                    fl = v.get("flops")
                    pa = v.get("param_size_mb")
                    valid_s = f"{valid:.2f}" if valid is not None else "n/a"
                    pred_s = f"{pred:.4f}" if pred is not None else "n/a"
                    fl_s = f"{fl:.2f}" if fl is not None else "n/a"
                    pa_s = f"{pa:.4f}" if pa is not None else "n/a"
                    print(f"  {k:>6}  {valid_s:>10}  {pred_s:>10}  {fl_s:>10}  {pa_s:>10}")

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
