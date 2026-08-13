"""Smoke tests for the LC-PFN scorer (needs the fetched checkpoint)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from fitness import FITNESS_SCORERS
from fitness.trace import TrainingTrace

CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'trained_models', 'lcpfn',
    'pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt')

needs_ckpt = pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason='LC-PFN checkpoint not fetched (scripts/fetch_lcpfn_checkpoint.sh)')


def make_trace(curve, epoch_len=196, interval=100):
    return TrainingTrace(
        minibatch_losses=[0.0] * (len(curve) * interval),
        val_acc_curve=list(curve),
        final_val_acc=curve[-1] if curve else None,
        epoch_len=epoch_len,
        snapshot_interval=interval,
    )


def saturating_curve(asymptote=0.9, rate=0.3, k=10):
    x = np.arange(1, k + 1)
    return list(asymptote - (asymptote - 0.1) * np.exp(-rate * x))


@needs_ckpt
def test_prediction_finite_in_unit_interval():
    scorer = FITNESS_SCORERS['lc_pfn'](ckpt_path=CKPT, target_epochs=20)
    pred = scorer.score(make_trace(saturating_curve()))
    assert np.isfinite(pred)
    assert 0.0 <= pred <= 1.0


@needs_ckpt
def test_extrapolation_at_least_tracks_curve_level():
    scorer = FITNESS_SCORERS['lc_pfn'](ckpt_path=CKPT, target_epochs=20)
    high = scorer.score(make_trace(saturating_curve(asymptote=0.92)))
    low = scorer.score(make_trace(saturating_curve(asymptote=0.55)))
    assert high > low


def test_below_min_observations_falls_back_to_early_stop(tmp_path):
    fake = tmp_path / 'fake.pt'
    fake.write_bytes(b'x')
    # 4 observations < 5: must return the last val acc without touching the model.
    scorer = FITNESS_SCORERS['lc_pfn'](ckpt_path=str(fake), target_epochs=20)
    pred = scorer.score(make_trace([0.2, 0.3, 0.35, 0.4]))
    assert pred == 0.4
