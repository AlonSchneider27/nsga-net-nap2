"""Zero-cost proxies: SynFlow, GradNorm, SNIP (NAP2 paper Tables 4-6).

Ports of the reference implementations benchmarked in White et al. 2021
(Abdelfattah et al.'s foresight-pruners measures, as vendored by NASLib).
Unlike the learning-curve scorers these consume no TrainingTrace: they score
the architecture AT INITIALIZATION from one minibatch (or, for SynFlow, from
an all-ones input with no data at all), so their score is independent of the
--nap2_steps budget. Higher = better for all three.

Each scorer deepcopies the model it is given, so gradient accumulation,
SynFlow's |w| rewrite, and BN running-stat updates never leak back into the
model that is about to be trained.
"""

import copy

import torch
import torch.nn as nn

from fitness.scorers import register_fitness


def _logits(out):
    """Tolerate tuple outputs; logits are the LAST element, matching nap2's
    _partial_train and fitness.trace (raw NB201 nets return (features, logits))."""
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


def _weight_grads(net):
    """(weight, grad) for Conv2d/Linear weights only — the reference's
    get_layer_metric_array(mode='param') scope. Biases and BN gamma/beta are
    excluded: their spatially-aggregated gradients are large relative to their
    element count and would shift rankings between architectures."""
    return [(m.weight, m.weight.grad) for m in net.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
            and m.weight is not None and m.weight.grad is not None]


def _grads_after_backward(model, inputs, targets):
    """Fresh private copy -> one CE forward/backward -> [(weight, grad)]."""
    net = copy.deepcopy(model)
    net.train()
    net.zero_grad()
    loss = nn.CrossEntropyLoss()(_logits(net(inputs)), targets)
    loss.backward()
    return _weight_grads(net)


@register_fitness('grad_norm')
class GradNorm:
    """Sum of per-parameter gradient L2 norms after one minibatch backward
    (Abdelfattah et al., 2021)."""
    needs_val_curve = False
    needs_final_val = False
    needs_init_model = True

    def score_init(self, model, inputs, targets):
        return float(sum(g.norm(2).item()
                         for _, g in _grads_after_backward(model, inputs, targets)))


@register_fitness('snip')
class SNIP:
    """Connection sensitivity sum |w * dL/dw| after one minibatch backward
    (Lee et al., 2018; single-shot saliency form)."""
    needs_val_curve = False
    needs_final_val = False
    needs_init_model = True

    def score_init(self, model, inputs, targets):
        return float(sum((p * g).abs().sum().item()
                         for p, g in _grads_after_backward(model, inputs, targets)))


@register_fitness('synflow')
class SynFlow:
    """Synaptic flow sum |w * dR/dw| (Tanaka et al., 2020): params replaced by
    their absolute values, forward on an all-ones input (data-free), backward
    on the summed output. BatchNorm is bypassed entirely (the reference scores
    a bn=False copy): live BN would re-normalize each layer's output and cancel
    the |w| path-product signal the measure is defined by. Computed in float64
    per the reference — without BN the flow grows multiplicatively with depth
    and overflows float32; on MPS (no float64 support) the private copy drops
    to CPU first.
    """
    needs_val_curve = False
    needs_final_val = False
    needs_init_model = True

    @staticmethod
    def _bypass(x):
        return x

    def score_init(self, model, inputs, targets):
        net = copy.deepcopy(model)
        device = next(net.parameters()).device
        if device.type == 'mps':
            net = net.cpu()
            device = torch.device('cpu')
        net = net.double()
        net.eval()
        net.zero_grad()
        for m in net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.forward = self._bypass
        with torch.no_grad():
            for p in net.parameters():
                p.data = p.data.abs()
        ones = torch.ones((1,) + tuple(inputs.shape[1:]),
                          dtype=torch.float64, device=device)
        torch.sum(_logits(net(ones))).backward()
        return float(sum((w * g).abs().sum().item()
                         for w, g in _weight_grads(net)))
