"""Verify NetworkCIFAR works on 16x16 ImageNet16-120 inputs.

AuxiliaryHeadCIFAR previously hardcoded an 8x8 input assumption (via
``AvgPool2d(5, stride=3)``). It was replaced with ``AdaptiveAvgPool2d((2,2))``
so the same architecture can run on CIFAR-10/100 (32x32 → 8x8 at the aux
hookpoint) and ImageNet16-120 (16x16 → 4x4).

Note on NetworkImageNet: its stem hardcodes three stride-2 convs designed
for 224x224 inputs, so it can't run on 16x16 regardless of pool changes.
The repo uses NetworkCIFAR (not NetworkImageNet) across all three datasets,
so NetworkImageNet on tiny inputs is intentionally out of scope.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from models import micro_genotypes
from models.micro_models import NetworkCIFAR


def _genotype():
    return micro_genotypes.NSGANet


@pytest.mark.parametrize('image_size,num_classes', [
    (32, 10),    # CIFAR-10
    (32, 100),   # CIFAR-100
    (16, 120),   # ImageNet16-120
])
def test_network_cifar_no_aux(image_size, num_classes):
    net = NetworkCIFAR(C=16, num_classes=num_classes, layers=8,
                       auxiliary=False, genotype=_genotype()).eval()
    net.droprate = 0.0
    out = net(torch.randn(2, 3, image_size, image_size))
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, num_classes)


@pytest.mark.parametrize('image_size,num_classes', [
    (32, 10),
    (32, 100),
    (16, 120),
])
def test_network_cifar_with_aux(image_size, num_classes):
    """AuxiliaryHeadCIFAR must accept whatever spatial the cells produce.

    Uses batch=2 because BatchNorm in training mode rejects single-element
    activations after the head's pool collapses spatial dims to 1x1.
    """
    net = NetworkCIFAR(C=16, num_classes=num_classes, layers=8,
                       auxiliary=True, genotype=_genotype()).train()
    net.droprate = 0.0
    out = net(torch.randn(2, 3, image_size, image_size))
    assert isinstance(out, tuple)
    logits, logits_aux = out
    assert logits.shape == (2, num_classes)
    if logits_aux is not None:
        assert logits_aux.shape == (2, num_classes)
