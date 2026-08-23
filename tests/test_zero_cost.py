"""Tests for the zero-cost proxies (synflow, grad_norm, snip)."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitness import ALL_BASELINES, build_scorers
from fitness.scorers import FITNESS_SCORERS


def tiny_net(num_classes=4):
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )


def batch(n=8):
    torch.manual_seed(11)
    return torch.randn(n, 3, 8, 8), torch.randint(0, 4, (n,))


def scorer(name):
    return FITNESS_SCORERS[name]()


def test_all_proxies_finite_and_positive():
    net = tiny_net()
    x, y = batch()
    for name in ('synflow', 'grad_norm', 'snip'):
        value = scorer(name).score_init(net, x, y)
        assert isinstance(value, float)
        assert value > 0
        assert value == value and value not in (float('inf'), float('-inf'))


def test_zeroed_weights_kill_all_scores():
    net = tiny_net()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    x, y = batch()
    # Weights-only sums: zero weights kill activations and weight gradients
    # alike (bias gradients exist but are out of scope by the reference).
    assert scorer('snip').score_init(net, x, y) == 0.0
    assert scorer('synflow').score_init(net, x, y) == 0.0
    assert scorer('grad_norm').score_init(net, x, y) == 0.0


def test_bias_and_bn_params_do_not_contribute():
    x, y = batch()
    net = tiny_net()
    before = {n: scorer(n).score_init(net, x, y)
              for n in ('synflow', 'grad_norm', 'snip')}
    with torch.no_grad():
        net[1].weight.mul_(5.0)    # BN gamma (bypassed by synflow)
        net[1].bias.add_(2.0)      # BN beta (bypassed by synflow)
        net[5].bias.add_(1.0)      # classifier bias (additive in R = sum logits)
    after = {n: scorer(n).score_init(net, x, y)
             for n in ('synflow', 'grad_norm', 'snip')}
    # synflow bypasses BN and sums conv/linear weights only, so it is exactly
    # invariant to these; grad_norm/snip run live BN on real data, so gamma
    # rescaling changes weight GRADIENTS — but the summed params themselves
    # stay weights-only, which is what the zeroed-weights test pins down.
    # (A CONV bias would legitimately change synflow — it feeds the forward
    # flow even though it is excluded from the sum; NB201 convs have none.)
    assert after['synflow'] == before['synflow']


def test_model_left_untouched():
    net = tiny_net()
    before = [p.detach().clone() for p in net.parameters()]
    x, y = batch()
    for name in ('synflow', 'grad_norm', 'snip'):
        scorer(name).score_init(net, x, y)
    for p, b in zip(net.parameters(), before):
        assert torch.equal(p, b)
    assert all(p.grad is None for p in net.parameters())


def test_tuple_output_uses_last_element():
    # Raw NB201 nets return (features, logits); logits are LAST.
    class TupleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = tiny_net()

        def forward(self, x):
            out = self.body(x)
            return (torch.zeros(out.shape[0], 999), out)

    x, y = batch()
    for name in ('synflow', 'grad_norm', 'snip'):
        assert scorer(name).score_init(TupleNet(), x, y) > 0


def test_registry_flags_and_all_order():
    for name in ('synflow', 'grad_norm', 'snip'):
        cls = FITNESS_SCORERS[name]
        assert cls.needs_init_model is True
        assert cls.needs_val_curve is False
        assert cls.needs_final_val is False
    assert ALL_BASELINES == ['synflow', 'grad_norm', 'snip',
                             'sotl', 'sotl_e', 'early_stop', 'lce_m', 'lc_pfn']


def test_build_scorers_zero_cost_only():
    names = [s.name for s in build_scorers('synflow,grad_norm,snip')]
    assert names == ['synflow', 'grad_norm', 'snip']
