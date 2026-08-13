"""Tests for scripts/lc_baselines_benchmark.py's trace prefixing."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

from lc_baselines_benchmark import prefix_trace
from fitness.trace import TrainingTrace


def full_trace(steps=3, interval=100, curve=None, final=None):
    return TrainingTrace(
        minibatch_losses=[0.1] * (steps * interval),
        val_acc_curve=list(curve) if curve else [],
        final_val_acc=final,
        epoch_len=196,
        snapshot_interval=interval,
    )


def test_prefix_with_curve():
    trace = full_trace(steps=3, curve=[0.2, 0.3, 0.4], final=0.4)
    sub = prefix_trace(trace, 2)
    assert len(sub.minibatch_losses) == 200
    assert sub.val_acc_curve == [0.2, 0.3]
    assert sub.final_val_acc == 0.3


def test_curve_free_mode_keeps_final_at_max_budget():
    # early_stop-only runs: no curve, but final_val_acc was measured at
    # budget end — the full-budget prefix must keep it.
    trace = full_trace(steps=3, final=0.55)
    sub = prefix_trace(trace, 3)
    assert sub.final_val_acc == 0.55


def test_curve_free_mode_smaller_budget_has_no_final():
    trace = full_trace(steps=3, final=0.55)
    sub = prefix_trace(trace, 2)
    assert sub.final_val_acc is None
