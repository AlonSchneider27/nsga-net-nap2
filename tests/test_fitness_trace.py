"""Tests for fitness/trace.py: run_partial_train emits a correct TrainingTrace."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fitness.trace import run_partial_train


def make_loader(n=24, batch=4, dim=8, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def test_trace_lengths_and_val_curve():
    model = nn.Linear(8, 3)
    train_q = make_loader()
    valid_q = make_loader(seed=1)
    trace = run_partial_train(model, train_q, valid_q, budget_minibatches=6,
                              need_val_curve=True,
                              training_config={'snapshot_interval': 3})
    assert len(trace.minibatch_losses) == 6
    assert len(trace.val_acc_curve) == 2          # boundaries at 3 and 6
    assert trace.final_val_acc == trace.val_acc_curve[-1]
    assert all(0.0 <= a <= 1.0 for a in trace.val_acc_curve)
    assert trace.epoch_len == len(train_q)
    assert trace.snapshot_interval == 3
    assert set(trace.times) == {'train', 'val'}


def test_no_val_passes_unless_requested():
    model = nn.Linear(8, 3)
    trace = run_partial_train(model, make_loader(), make_loader(seed=1),
                              budget_minibatches=4,
                              training_config={'snapshot_interval': 2})
    assert trace.val_acc_curve == []
    assert trace.final_val_acc is None
    assert trace.times['val'] == 0.0


def test_final_val_only():
    model = nn.Linear(8, 3)
    trace = run_partial_train(model, make_loader(), make_loader(seed=1),
                              budget_minibatches=4, need_final_val=True,
                              training_config={'snapshot_interval': 2})
    assert trace.val_acc_curve == []
    assert 0.0 <= trace.final_val_acc <= 1.0


def test_training_actually_updates_the_given_model():
    # The runner trains the model it is given (callers pass a throwaway copy).
    model = nn.Linear(8, 3)
    before = model.weight.detach().clone()
    run_partial_train(model, make_loader(), make_loader(seed=1),
                      budget_minibatches=3,
                      training_config={'snapshot_interval': 100})
    assert not torch.equal(before, model.weight.detach())


def test_epoch_native_scores_values():
    from fitness.trace import epoch_native_scores
    from fitness.scorers import FITNESS_SCORERS
    scorers = [FITNESS_SCORERS[n]() for n in ('sotl', 'sotl_e', 'early_stop')]
    loss_sums = [100.0, 60.0, 40.0]          # per-epoch summed minibatch losses
    val_accs = [0.30, 0.45, 0.55]
    s = epoch_native_scores(scorers, loss_sums, val_accs)
    # sotl@eK = -(sum of epoch sums 1..K)
    assert s['sotl@e1'] == -100.0
    assert s['sotl@e3'] == -200.0
    # sotl_e@eK = -(epoch K's sum) — NASLib's per-epoch native definition
    assert s['sotl_e@e1'] == -100.0
    assert s['sotl_e@e3'] == -40.0
    # early_stop@eK = val acc at epoch K
    assert s['early_stop@e2'] == 0.45
    assert len(s) == 9                        # 3 methods x 3 epochs


def test_epoch_native_scores_nonfinite_dropped():
    from fitness.trace import epoch_native_scores
    from fitness.scorers import FITNESS_SCORERS
    s = epoch_native_scores([FITNESS_SCORERS['sotl']()],
                            [float('nan'), 50.0], [])
    assert s['sotl@e1'] is None               # NaN epoch sum -> dropped
