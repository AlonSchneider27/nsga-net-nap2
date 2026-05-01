"""NAS-Bench-201 operations and model architecture.

Standalone implementation — no xautodl dependency required.
Replicates the exact architecture from xautodl to produce identical models.

Source: xautodl/models/cell_operations.py and cell_infers/tiny_network.py
Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019
"""

import torch
import torch.nn as nn
from copy import deepcopy


# ============================================================
# Operations used in NAS-Bench-201
# ============================================================

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
                 affine, track_running_stats=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=not affine),
            nn.BatchNorm2d(C_out, affine=affine,
                           track_running_stats=track_running_stats),
        )

    def forward(self, x):
        return self.op(x)


class POOLING(nn.Module):
    def __init__(self, C_in, C_out, stride, mode, affine=True,
                 track_running_stats=True):
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1,
                                   count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Invalid mode={mode} in POOLING")

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            return x.new_zeros(shape, dtype=x.dtype, device=x.device)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super().__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0,
                              bias=not affine))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0,
                                  bias=not affine)
        else:
            raise ValueError(f"Invalid stride: {stride}")
        self.bn = nn.BatchNorm2d(C_out, affine=affine,
                                  track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])],
                            dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True,
                 track_running_stats=True):
        super().__init__()
        assert stride in (1, 2), f"invalid stride {stride}"
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine,
                                  track_running_stats)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine,
                                  track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0,
                          bias=False),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine,
                                          track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


# ============================================================
# Operations registry (NAS-Bench-201 search space)
# ============================================================

OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats:
        Zero(C_in, C_out, stride),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats:
        POOLING(C_in, C_out, stride, "avg", affine, track_running_stats),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats:
        ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1),
                   affine, track_running_stats),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats:
        ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1),
                   affine, track_running_stats),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats:
        Identity() if stride == 1 and C_in == C_out
        else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}


# ============================================================
# Cell and Network definitions
# ============================================================

class InferCell(nn.Module):
    """A single cell in the NAS-Bench-201 network."""

    def __init__(self, genotype, C_in, C_out, stride, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)

        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](C_in, C_out, stride, affine,
                                         track_running_stats)
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine,
                                         track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(
                zip(self.node_IX, self.node_IN)):
            node_feature = sum(
                self.layers[_il](nodes[_ii])
                for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]


class TinyNetwork(nn.Module):
    """NAS-Bench-201 macro architecture.

    Structure: stem -> [N cells, reduction, N cells, reduction, N cells] -> classifier
    """

    def __init__(self, C, N, genotype, num_classes):
        super().__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


# ============================================================
# Architecture string parsing
# ============================================================

def parse_arch_str(arch_str):
    """Parse NAS-Bench-201 architecture string into genotype.

    The genotype includes an implicit input node at index 0 (no operations).
    InferCell iterates range(1, len(genotype)), so len must be num_op_nodes + 1.

    Args:
        arch_str: e.g. '|avg_pool_3x3~0|+|avg_pool_3x3~0|avg_pool_3x3~1|+|none~0|skip_connect~1|avg_pool_3x3~2|'

    Returns: tuple of tuples with input node prepended, e.g.:
        (
            (('avg_pool_3x3', 0),),          # ops for node 1
            (('avg_pool_3x3', 0), ('avg_pool_3x3', 1)),  # ops for node 2
            (('none', 0), ('skip_connect', 1), ('avg_pool_3x3', 2)),  # ops for node 3
            (),  # sentinel — InferCell uses range(1, len) so this makes len=4
        )
    """
    node_strs = arch_str.split('+')
    genotypes = []
    for node_str in node_strs:
        inputs = [x for x in node_str.split('|') if x != '']
        inputs = tuple(
            (op.split('~')[0], int(op.split('~')[1])) for op in inputs
        )
        genotypes.append(inputs)
    # Append sentinel node so len(genotype) = num_op_nodes + 1
    # This matches xautodl's CellStructure which has nodes=len(genotype)
    # and InferCell iterates range(1, len(genotype))
    genotypes.append(())
    return tuple(genotypes)


def build_nb201_model(arch_str, num_classes=10, C=16, N=5):
    """Build a NAS-Bench-201 model from architecture string.

    Args:
        arch_str: NAS-Bench-201 architecture topology string
        num_classes: number of output classes (10, 100, or 120)
        C: base channels (default: 16)
        N: number of cells per stage (default: 5)

    Returns: TinyNetwork nn.Module
    """
    genotype = parse_arch_str(arch_str)
    return TinyNetwork(C, N, genotype, num_classes)
