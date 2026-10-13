"""Shared partial-training trace for the learning-curve fitness baselines.

One short training run per candidate produces a TrainingTrace; every selected
scorer (SoTL, SoTL-E, Early-Stop, LCE-m, LC-PFN) reads from it. The training
dynamics deliberately mirror nap2's ``NAP2Predictor._partial_train`` (same SGD
config, same infinite reshuffle iterator, no grad clipping, no LR scheduler)
so baseline scores and nap2's pred_acc see the same optimization trajectory at
the same budget.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from nap2.predictor import DEFAULT_TRAINING_CONFIG


@dataclass
class TrainingTrace:
    minibatch_losses: List[float]
    val_acc_curve: List[float]          # [0,1] at each snapshot boundary; [] unless requested
    final_val_acc: Optional[float]      # [0,1] at budget end; None unless requested
    epoch_len: int                      # len(train_queue) at runtime
    snapshot_interval: int
    times: Dict[str, float] = field(default_factory=dict)   # {'train': s, 'val': s}


@torch.no_grad()
def _eval_acc(model, valid_queue, device):
    """Validation accuracy in [0,1]; restores the model's training mode."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in valid_queue:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        # NAS-Bench-201 models return (features, logits) — same convention
        # nap2's _partial_train assumes.
        if isinstance(outputs, tuple):
            outputs = outputs[-1]
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if was_training:
        model.train()
    return correct / total


def run_partial_train(model, train_queue, valid_queue, budget_minibatches,
                      need_val_curve=False, need_final_val=False,
                      training_config=None):
    """Train ``model`` for ``budget_minibatches`` and record a TrainingTrace.

    Validation passes are only run when a selected scorer needs them:
    ``need_val_curve`` evaluates at every snapshot boundary (expensive),
    ``need_final_val`` alone evaluates once at the end.
    The caller is responsible for passing a throwaway copy of the model.
    """
    config = dict(DEFAULT_TRAINING_CONFIG)
    if training_config:
        config.update(training_config)
    interval = config["snapshot_interval"]

    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        nesterov=config["nesterov"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    minibatch_losses = []
    val_acc_curve = []
    t_train = 0.0
    t_val = 0.0

    batch_count = 0
    data_iter = iter(train_queue)
    while batch_count < budget_minibatches:
        t0 = time.perf_counter()
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_queue)
            inputs, targets = next(data_iter)

        inputs = inputs.to(model_device, non_blocking=True)
        targets = targets.to(model_device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[-1]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # .item() synchronizes the device, so the MPS async-capture hazard
        # nap2's _partial_train guards against does not arise here.
        minibatch_losses.append(loss.item())
        batch_count += 1
        t_train += time.perf_counter() - t0

        if need_val_curve and batch_count % interval == 0:
            t0 = time.perf_counter()
            val_acc_curve.append(_eval_acc(model, valid_queue, model_device))
            t_val += time.perf_counter() - t0

    final_val_acc = None
    if need_final_val or need_val_curve:
        if val_acc_curve and budget_minibatches % interval == 0:
            final_val_acc = val_acc_curve[-1]
        elif need_final_val:
            t0 = time.perf_counter()
            final_val_acc = _eval_acc(model, valid_queue, model_device)
            t_val += time.perf_counter() - t0

    return TrainingTrace(
        minibatch_losses=minibatch_losses,
        val_acc_curve=val_acc_curve,
        final_val_acc=final_val_acc,
        epoch_len=len(train_queue),
        snapshot_interval=interval,
        times={'train': t_train, 'val': t_val},
    )


def prefix_trace(trace, steps):
    """Truncate a trace to a smaller snapshot budget (prefix of the run).

    The paper's budget protocol: every method at budget k consumes the first
    k*interval minibatches of ONE partial-training run. In curve-free mode
    (no val curve collected) only the full budget keeps its final_val_acc.
    """
    mb = steps * trace.snapshot_interval
    curve = trace.val_acc_curve[:steps]
    if curve:
        final = curve[-1]
    elif mb == len(trace.minibatch_losses):
        final = trace.final_val_acc
    else:
        final = None
    return TrainingTrace(
        minibatch_losses=trace.minibatch_losses[:mb],
        val_acc_curve=curve,
        final_val_acc=final,
        epoch_len=trace.epoch_len,
        snapshot_interval=trace.snapshot_interval,
        times=trace.times,
    )


def epoch_native_scores(scorers, epoch_loss_sums, epoch_val_accs,
                        target_suffix='e'):
    """Score trace-scorers at their NATIVE epoch cadence.

    Inputs come from the REAL training loop: per-epoch summed minibatch
    train losses, and val accuracy [0,1] measured at each epoch boundary
    (empty list if no val-based scorer is selected). Each budget K in
    epochs 1..E is scored from a pseudo-trace where one "minibatch" is one
    epoch (epoch_len=1, snapshot_interval=1): sotl = -(sum of epoch sums),
    sotl_e = -(last epoch sum) — NASLib's per-epoch definitions — and the
    extrapolators see x = 1..K epochs with x_target = their epoch horizon.

    Returns {f"{name}@{target_suffix}{K}": float|None}.
    """
    scores = {}
    n_epochs = len(epoch_loss_sums)
    full = TrainingTrace(
        minibatch_losses=list(epoch_loss_sums),
        val_acc_curve=list(epoch_val_accs),
        final_val_acc=epoch_val_accs[-1] if epoch_val_accs else None,
        epoch_len=1,
        snapshot_interval=1,
        times={'train': 0.0, 'val': 0.0},
    )
    for scorer in scorers:
        for k in range(1, n_epochs + 1):
            key = f'{scorer.name}@{target_suffix}{k}'
            try:
                value = float(scorer.score(prefix_trace(full, k)))
                if value != value or value in (float('inf'), float('-inf')):
                    value = None
                scores[key] = value
            except Exception:
                import logging
                logging.exception('epoch-native %s failed at epoch %d',
                                  scorer.name, k)
                scores[key] = None
    return scores
