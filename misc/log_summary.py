"""Scrape per-architecture metrics from an NSGA-Net + nap2 search log.

The search log written by ``search/evolution_search.py`` emits, for each
evaluated architecture, a fixed sequence of lines::

    ... Network id = <id>
    ... Architecture = Genotype(normal=[...], normal_concat=[...], reduce=[...], reduce_concat=[...])
    ... param size = <float>MB
    ...                                            (training output)
    ... flops = <float>
    ... arch <id>: valid_acc=<float> pred_acc=<float|n/a>

This module reads such a log and emits a dict mapping arch id (str) to::

    {
        "valid_acc":      float,
        "pred_acc":       float | None,    # None when the predictor failed
        "flops":          float,
        "param_size_mb":  float,
        "genotype":       str,             # verbatim Genotype(...) repr
    }

``write_summary`` additionally computes a top-level rollup of ranking
metrics (Kendall tau, Spearman rho, top-10% overlap) between
predicted and observed accuracy across all architectures with both
signals, and packages the result as::

    {
        "architectures": {arch_id: {...}, ...},
        "metrics":       {kendall_tau, spearman_rho, ...},
    }

The scraper is the importable core; ``scripts/summarize_search.py`` is a
thin CLI on top.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional, Union


# ----------------------------- regex patterns -----------------------------

NETWORK_ID_RE = re.compile(r"Network id\s*=\s*(\d+)")
# Capture the full Genotype(...) or NB201Genotype(...) repr including
# its closing paren. The repr is single-line in our logs, so a greedy
# match to the end of the line is correct.
GENOTYPE_RE = re.compile(
    r"Architecture\s*=\s*((?:NB201Genotype|Genotype)\(.*\))\s*$"
)
PARAM_SIZE_RE = re.compile(r"param size\s*=\s*([0-9.eE+-]+)\s*MB")
FLOPS_RE = re.compile(r"flops\s*=\s*([0-9.eE+-]+)")
ARCH_SUMMARY_RE = re.compile(
    r"arch\s+(\d+):\s*valid_acc=([0-9.eE+-]+)\s+pred_acc=(n/a|[0-9.eE+-]+)"
)


# ----------------------------- public API ---------------------------------

def scrape(log_path: Union[str, Path]) -> Dict[str, Dict]:
    """Parse one search log and return a dict of per-arch metrics.

    Args:
        log_path: path to the run's ``log.txt`` file.

    Returns:
        Mapping arch_id (str) -> {valid_acc, pred_acc, flops,
        param_size_mb, genotype}. ``pred_acc`` is ``None`` when the
        predictor failed; the other fields are required and are populated
        from the per-architecture buffer right before the
        ``arch N: ...`` summary line.
    """
    log_path = Path(log_path)
    results: Dict[str, Dict] = {}

    # Per-architecture buffer. Reset after each arch summary is flushed.
    buf_network_id: Optional[int] = None
    buf_genotype: Optional[str] = None
    buf_param_size: Optional[float] = None
    buf_flops: Optional[float] = None

    with log_path.open("r", errors="replace") as f:
        for line in f:
            m = NETWORK_ID_RE.search(line)
            if m:
                buf_network_id = int(m.group(1))
                buf_genotype = None
                buf_param_size = None
                buf_flops = None
                continue

            m = GENOTYPE_RE.search(line)
            if m:
                buf_genotype = m.group(1).strip()
                continue

            m = PARAM_SIZE_RE.search(line)
            if m:
                buf_param_size = float(m.group(1))
                continue

            m = FLOPS_RE.search(line)
            if m:
                buf_flops = float(m.group(1))
                continue

            m = ARCH_SUMMARY_RE.search(line)
            if m:
                arch_id = int(m.group(1))
                if buf_network_id is not None and buf_network_id != arch_id:
                    logging.warning(
                        "log_summary: arch summary id %d does not match the most "
                        "recent 'Network id = %d' line; buffer may be stale.",
                        arch_id, buf_network_id,
                    )

                pred_token = m.group(3)
                pred_acc: Optional[float] = (
                    None if pred_token == "n/a" else float(pred_token)
                )

                results[str(arch_id)] = {
                    "valid_acc": float(m.group(2)),
                    "pred_acc": pred_acc,
                    "flops": buf_flops,
                    "param_size_mb": buf_param_size,
                    "genotype": buf_genotype,
                }

                # Reset buffer; the next arch starts with a fresh
                # Network id / Architecture / param size sequence.
                buf_network_id = None
                buf_genotype = None
                buf_param_size = None
                buf_flops = None

    # Stable, numeric-id ordering for human-readable JSON.
    return {k: results[k] for k in sorted(results, key=int)}


def compute_run_metrics(architectures: Dict[str, Dict]) -> Dict[str, object]:
    """Compute ranking metrics over a scraped architectures dict.

    Calls into :func:`nap2.training.evaluate.compute_metrics` over the
    architectures that have **both** ``valid_acc`` and a non-None
    ``pred_acc``. Architectures with ``pred_acc=None`` (predictor
    failures) are tallied in ``num_failed_predictions`` and excluded
    from the correlation.

    Returns a dict with keys ``kendall_tau``, ``spearman_rho``,
    ``top_10pct_accuracy``, ``num_architectures``,
    ``num_failed_predictions``. The first three are floats, or
    ``None`` if there are fewer than 2 paired observations (not enough
    for a correlation) or if the scipy result is NaN. JSON-serializable.
    """
    n_failed = sum(
        1 for v in architectures.values() if v.get("pred_acc") is None
    )

    predicted = {
        k: v["pred_acc"]
        for k, v in architectures.items()
        if v.get("pred_acc") is not None and v.get("valid_acc") is not None
    }
    ground_truth = {
        k: v["valid_acc"]
        for k, v in architectures.items()
        if v.get("valid_acc") is not None
    }

    n_paired = len(set(predicted) & set(ground_truth))
    if n_paired < 2:
        return {
            "kendall_tau": None,
            "spearman_rho": None,
            "top_10pct_accuracy": None,
            "num_architectures": n_paired,
            "num_failed_predictions": n_failed,
        }

    # Lazy import so log_summary stays usable in environments where
    # nap2 isn't importable (e.g. the scraper-only CLI).
    from nap2.training.evaluate import compute_metrics

    raw = compute_metrics(predicted, ground_truth)

    def _clean(x):
        # scipy returns NaN when one side is constant; JSON can't
        # encode NaN portably, so coerce to None.
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    return {
        "kendall_tau": _clean(raw.get("kendall_tau")),
        "spearman_rho": _clean(raw.get("spearman_rho")),
        "top_10pct_accuracy": _clean(raw.get("top_10pct_accuracy")),
        "num_architectures": int(raw.get("num_architectures", n_paired)),
        "num_failed_predictions": n_failed,
    }


def write_summary(
    log_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Dict[str, object]:
    """Scrape ``log_path`` and write a JSON summary to ``output_path``.

    The on-disk shape is::

        {
            "architectures": {arch_id: {...}, ...},
            "metrics":       {kendall_tau, spearman_rho, ...}
        }

    Returns the same dict so callers can act on it without re-reading
    the file. ``output_path``'s parent directory must already exist.

    If metrics computation fails for any reason (e.g. nap2 import
    error in a stripped-down env), the metrics block is replaced with
    ``{"error": "<message>"}`` so the architectures payload still
    gets written.
    """
    architectures = scrape(log_path)
    try:
        metrics = compute_run_metrics(architectures)
    except Exception as e:
        logging.exception("compute_run_metrics failed; emitting error in summary")
        metrics = {"error": f"{type(e).__name__}: {e}"}

    payload = {"architectures": architectures, "metrics": metrics}
    with Path(output_path).open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def resolve_log_path(input_path: Union[str, Path]) -> Path:
    """Accept either a run directory or a log file; return the log file."""
    p = Path(input_path)
    if p.is_dir():
        candidate = p / "log.txt"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{p} is a directory but no log.txt found inside it"
            )
        return candidate
    if p.is_file():
        return p
    raise FileNotFoundError(f"{p} is neither a file nor a directory")
