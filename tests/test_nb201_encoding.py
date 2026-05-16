"""Tests for the new NB201 search-space encoding.

The encoding maps a length-6 integer vector into an NB201 arch_str via
``search/nb201_encoding.py``. These tests cover:

  - convert() identity
  - decode() shape + canonical-form sanity
  - decode() raises on malformed input
  - the produced arch_str round-trips through nap2's parse_arch_str
  - end-to-end: feed the arch_str into build_nb201_model and run a
    forward pass at both 32x32 (CIFAR) and 16x16 (ImageNet16-120) inputs
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from models.micro_genotypes import NB201_PRIMITIVES, NB201Genotype
from nap2.search_spaces.nb201_ops import build_nb201_model, parse_arch_str
from search import nb201_encoding


def test_convert_is_identity_on_in_range_ints():
    assert nb201_encoding.convert([0, 1, 2, 3, 4, 0]) == [0, 1, 2, 3, 4, 0]


def test_convert_accepts_numpy_like_array():
    import numpy as np
    arr = np.array([0, 1, 2, 3, 4, 0])
    assert nb201_encoding.convert(arr) == [0, 1, 2, 3, 4, 0]


def test_decode_returns_nb201_genotype():
    g = nb201_encoding.decode([3, 3, 3, 3, 3, 3])  # all nor_conv_3x3
    assert isinstance(g, NB201Genotype)
    assert "nor_conv_3x3" in g.arch_str
    # Three nodes worth of edges separated by '+'.
    assert g.arch_str.count("+") == 2


def test_decode_all_nor_conv_3x3_is_canonical():
    g = nb201_encoding.decode([3, 3, 3, 3, 3, 3])
    expected = (
        "|nor_conv_3x3~0|+"
        "|nor_conv_3x3~0|nor_conv_3x3~1|+"
        "|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|"
    )
    assert g.arch_str == expected


def test_decode_distinct_ops_are_placed_in_correct_edges():
    # Pick a different op for each edge so we can read it off.
    g = nb201_encoding.decode([0, 1, 2, 3, 4, 0])
    # Order: edges to node 1 (1), node 2 (2), node 3 (3).
    expected = (
        f"|{NB201_PRIMITIVES[0]}~0|+"
        f"|{NB201_PRIMITIVES[1]}~0|{NB201_PRIMITIVES[2]}~1|+"
        f"|{NB201_PRIMITIVES[3]}~0|{NB201_PRIMITIVES[4]}~1|{NB201_PRIMITIVES[0]}~2|"
    )
    assert g.arch_str == expected


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError, match="6 entries"):
        nb201_encoding.decode([0, 0, 0, 0, 0])


def test_decode_rejects_out_of_range_op_indices():
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        nb201_encoding.decode([0, 0, 0, 0, 0, 5])


def test_decode_arch_str_roundtrips_through_parse_arch_str():
    """Whatever decode emits must be acceptable to nap2's NB201 parser."""
    g = nb201_encoding.decode([3, 2, 1, 4, 0, 3])
    # parse_arch_str returns a tuple-of-tuples internal genotype; just
    # check it doesn't raise.
    parsed = parse_arch_str(g.arch_str)
    assert parsed is not None
    # Three op-nodes (plus an implicit input at index 0); see nb201_ops.parse_arch_str.
    assert len(parsed) >= 3


@pytest.mark.parametrize("image_size,num_classes", [
    (32, 10),    # CIFAR-10
    (32, 100),   # CIFAR-100
    (16, 120),   # ImageNet16-120
])
def test_end_to_end_build_nb201_model_runs_forward(image_size, num_classes):
    """Decode -> build_nb201_model -> forward should produce logits of the
    expected shape on each supported dataset's input geometry."""
    g = nb201_encoding.decode([3, 1, 3, 1, 3, 3])  # mix of skip + conv
    model = build_nb201_model(
        g.arch_str,
        num_classes=num_classes,
        C=16,
        N=1,  # tiny so the test is fast
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, image_size, image_size))
    # TinyNetwork.forward returns (features, logits); we want logits.
    assert isinstance(out, tuple)
    features, logits = out
    assert logits.shape == (2, num_classes)
