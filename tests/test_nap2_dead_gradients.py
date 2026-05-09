"""Verify nap2's stats pipeline tolerates Conv2d layers with no gradient.

Background
----------

In a 620-architecture CIFAR-100 search we observed ~50 ``nap2 prediction
failed`` lines, all with the same traceback:

    File "nap2/stats.py", line 115, in extract_layer_stats
        data = np.where(np.isinf(data), np.nan, data)
    TypeError: ufunc 'isinf' not supported for the input types ...

Root cause: SnapshotCollector stores Python ``None`` when a Conv2d's
``module.weight.grad`` is None. Heavy-pool / narrow-concat genotypes
have entire sub-graphs that receive no gradient flow during nap2's
partial-training mini-batch, so ``weight.grad`` stays None. ``np.isinf``
on the resulting object-dtype array raises.

Fix: ``extract_all_stats`` skips ``None`` tensors. ``extract_layer_stats``
gained a defensive guard for direct callers.

This test exercises both behaviors using only nap2's own code (no
checkpoint files needed).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from nap2.stats import extract_all_stats, extract_layer_stats


def test_extract_all_stats_skips_none_tensor():
    """The bug from the production search: a None tensor in a snapshot."""
    snapshots = {
        0: {
            "alive_layer/conv2d/kernel": np.random.randn(3, 3, 3, 3).astype(np.float64),
            "dead_layer/conv2d/kernel": None,
        }
    }
    out = extract_all_stats(snapshots)
    assert "alive_layer/conv2d/kernel" in out[0]
    assert "dead_layer/conv2d/kernel" not in out[0]


def test_extract_all_stats_handles_all_dead_layers():
    """If every layer is None, the step result is just an empty dict (no crash)."""
    snapshots = {
        0: {
            "dead_a/conv2d/kernel": None,
            "dead_b/conv2d/kernel": None,
        }
    }
    out = extract_all_stats(snapshots)
    assert out == {0: {}}


def test_extract_layer_stats_rejects_none_directly():
    with pytest.raises(ValueError, match="tensor is None"):
        extract_layer_stats(None)


def test_extract_layer_stats_rejects_object_array():
    obj_arr = np.array([None, None], dtype=object)
    with pytest.raises(ValueError, match="object-dtype"):
        extract_layer_stats(obj_arr)


def test_extract_layer_stats_still_works_on_real_tensor():
    """Sanity: the guard doesn't break the happy path."""
    tensor = np.random.randn(8, 4, 3, 3).astype(np.float64)
    stats = extract_layer_stats(tensor)
    # We don't pin the exact key set (it's nap2 internal); just that we got
    # something back and it's a dict of finite values.
    assert isinstance(stats, dict)
    assert len(stats) > 0
