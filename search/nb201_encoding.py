"""NAS-Bench-201 encoding for the NSGA-Net GA.

Maps a length-6 integer vector (the pymoo decision-variable form) to and
from the canonical NB201 arch_str. Public API mirrors
``search/micro_encoding.py`` so the dispatch in ``search/train_search.py``
stays uniform across search spaces.

The encoding layout (6 edges in canonical NB201 order):

    genome[0] -> edge (0 -> 1)             # node 1 has 1 incoming edge
    genome[1] -> edge (0 -> 2)             # node 2 has 2 incoming edges
    genome[2] -> edge (1 -> 2)
    genome[3] -> edge (0 -> 3)             # node 3 has 3 incoming edges
    genome[4] -> edge (1 -> 3)
    genome[5] -> edge (2 -> 3)

Each genome entry is an index into ``NB201_PRIMITIVES``
(``models.micro_genotypes``). The 5**6 = 15,625 combinations correspond
1:1 with the entries in the published NB201 catalog.
"""

from __future__ import annotations

from typing import List, Sequence

from models.micro_genotypes import NB201_PRIMITIVES, NB201Genotype


N_EDGES = 6


def convert(bit_string: Sequence) -> List[int]:
    """pymoo integer array (length 6) -> list[int] (op indices)."""
    return [int(v) for v in bit_string]


def decode(genome: Sequence[int]) -> NB201Genotype:
    """list[int] of length 6 -> NB201Genotype(arch_str=...).

    Produces the exact arch_str format used by NB201 / NATS-Bench, e.g.

        |nor_conv_3x3~0|+|skip_connect~0|nor_conv_3x3~1|+|none~0|skip_connect~1|avg_pool_3x3~2|
    """
    if len(genome) != N_EDGES:
        raise ValueError(
            f"NB201 genome must have {N_EDGES} entries, got {len(genome)}"
        )
    if any(int(v) < 0 or int(v) >= len(NB201_PRIMITIVES) for v in genome):
        raise ValueError(
            f"NB201 genome entries must be in [0, {len(NB201_PRIMITIVES) - 1}], "
            f"got {list(genome)}"
        )
    ops = [NB201_PRIMITIVES[int(i)] for i in genome]
    # Node 1: 1 edge from node 0.
    # Node 2: 2 edges from nodes 0, 1.
    # Node 3: 3 edges from nodes 0, 1, 2.
    node1 = f"|{ops[0]}~0|"
    node2 = f"|{ops[1]}~0|{ops[2]}~1|"
    node3 = f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    return NB201Genotype(arch_str=f"{node1}+{node2}+{node3}")
