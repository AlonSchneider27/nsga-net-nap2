from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

ENAS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_5x5', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('avg_pool_3x3', 0)
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 4),
        ('avg_pool_3x3', 1),
        ('sep_conv_3x3', 5),
        ('sep_conv_5x5', 0)
    ],
    reduce_concat=[2, 3, 6]
)

NSGANet = Genotype(
    normal=[
        ('skip_connect', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 1),
        ('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 3),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=[2, 4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('dil_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('dil_conv_5x5', 1),
        ('skip_connect', 2),
        ('avg_pool_3x3', 1),
        ('dil_conv_5x5', 1),
        ('dil_conv_3x3', 1)
    ],
    reduce_concat=[3, 4, 5, 6]
)


# NAS-Bench-201 ----------------------------------------------------------
# NB201 cells: 4 nodes, 6 directed edges (edges to node 1, 2, 3 with
# 1+2+3=6 op slots). Each edge picks one of these 5 primitives. The
# design space is 5**6 = 15,625 architectures, exactly matching the
# published NB201 catalog — every architecture the GA can sample is a
# known NB201 entry, so the user can cross-reference summary.json
# offline against the NB201 catalog without any API installation.
#
# See nap2/search_spaces/nb201_ops.py for the actual op implementations.
NB201_PRIMITIVES = [
    'none',
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
    'avg_pool_3x3',
]

# Single-field container so summary.json / logs see a clean arch_str.
# Round-trippable via nap2.search_spaces.nb201_ops.parse_arch_str.
NB201Genotype = namedtuple('NB201Genotype', 'arch_str')
