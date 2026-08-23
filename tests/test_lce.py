"""Tests for the LCE-m parametric ensemble (fitness/lce.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fitness import FITNESS_SCORERS
from fitness.trace import TrainingTrace


def make_trace(curve, epoch_len=196, interval=100):
    return TrainingTrace(
        minibatch_losses=[0.0] * (len(curve) * interval),
        val_acc_curve=list(curve),
        final_val_acc=curve[-1] if curve else None,
        epoch_len=epoch_len,
        snapshot_interval=interval,
    )


def saturating_curve(asymptote, rate=0.3, k=10, noise=0.005, seed=0):
    rng = np.random.RandomState(seed)
    x = np.arange(1, k + 1)
    return list(asymptote - (asymptote - 0.1) * np.exp(-rate * x)
                + rng.randn(k) * noise)


def test_prediction_is_finite_and_sane():
    np.random.seed(0)
    scorer = FITNESS_SCORERS['lce_m'](target_epochs=20, mcmc_steps=50)
    curve = saturating_curve(0.9)
    pred = scorer.score(make_trace(curve))
    assert np.isfinite(pred)
    # Extrapolation of an increasing curve should not fall far below the last
    # observation, nor explode.
    assert curve[-1] - 0.15 < pred < 1.5


def test_ranks_two_curves_correctly():
    np.random.seed(0)
    scorer = FITNESS_SCORERS['lce_m'](target_epochs=20, mcmc_steps=50)
    high = scorer.score(make_trace(saturating_curve(0.92, seed=1)))
    low = scorer.score(make_trace(saturating_curve(0.55, seed=2)))
    assert high > low


def test_mcmc_failure_hits_fallback(monkeypatch):
    import fitness.lce as lce_mod
    np.random.seed(0)

    def boom(self, y, N=300, var=0.0001):
        raise RuntimeError('forced failure')

    monkeypatch.setattr(lce_mod.ParametricEnsemble, 'mcmc', boom)
    scorer = FITNESS_SCORERS['lce_m'](target_epochs=20, mcmc_steps=20)
    pred = scorer.score(make_trace([0.5] * 6))
    # NASLib's default-guess-plus-jitter fallback on the [0,1] scale.
    assert 0.85 <= pred < 0.87
