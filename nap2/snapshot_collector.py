"""Snapshot Collector: captures weights and gradients during neural network training.

Captures Conv2d and Linear layer parameters at regular intervals (every N mini-batches)
as the first stage of NAP2's pipeline.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple, Type

import numpy as np
import torch.nn as nn


class SnapshotCollector:
    """Captures weight and gradient snapshots from Conv2d/Linear layers during training.

    Args:
        model: PyTorch model to monitor.
        layer_types: Tuple of layer types to capture (default: Conv2d, Linear).
        interval: Number of mini-batches between snapshots.
        max_snapshots: Maximum number of snapshots to capture.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_types: Tuple[Type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
        interval: int = 100,
        max_snapshots: int = 23,
    ) -> None:
        self._model = model
        self._layer_types = layer_types
        self._interval = interval
        self._max_snapshots = max_snapshots
        self._batch_number = 0
        self._snapshots: OrderedDict[int, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Call after each mini-batch (after backward + optimizer step)."""
        self._batch_number += 1
        if (
            self._batch_number % self._interval == 0
            and len(self._snapshots) < self._max_snapshots
        ):
            self._capture()

    def get_snapshots(self) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """Return all snapshots.

        Returns:
            ``{batch_num: {layer_name: {"weights": ndarray, "gradients": ndarray}}}``
        """
        return dict(self._snapshots)

    def get_weight_snapshots(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Return weight-only snapshots.

        Returns:
            ``{batch_num: {layer_name: ndarray}}``
        """
        return {
            batch: {name: data["weights"] for name, data in layers.items()}
            for batch, layers in self._snapshots.items()
        }

    def get_gradient_snapshots(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Return gradient-only snapshots.

        Returns:
            ``{batch_num: {layer_name: ndarray}}``
        """
        return {
            batch: {name: data["gradients"] for name, data in layers.items()}
            for batch, layers in self._snapshots.items()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture(self) -> None:
        """Capture a single snapshot of all tracked layers.

        A dead layer (no gradient flow this batch — e.g. its output is zeroed
        by NB201 ``none`` edges) leaves ``.grad`` as None. Store an explicit
        zero gradient instead of None: it keeps the layer at its position in
        the fixed 65-slot feature-map grid (dropping it would shift every
        later layer up), and a zero-stats row is the "dead layer" signature
        the predictor's training data contains for such architectures.
        """
        snapshot: Dict[str, Dict[str, np.ndarray]] = {}

        def _grad_or_zeros(param) -> np.ndarray:
            if param.grad is not None:
                return param.grad.cpu().numpy().copy()
            return np.zeros_like(param.data.cpu().numpy())

        for name, module in self._model.named_modules():
            if not isinstance(module, tuple(self._layer_types)):
                continue

            if isinstance(module, nn.Conv2d):
                key = f"{name}/conv2d/kernel"
                snapshot[key] = {
                    "weights": module.weight.data.cpu().numpy().copy(),
                    "gradients": _grad_or_zeros(module.weight),
                }

            elif isinstance(module, nn.Linear):
                # Weight
                w_key = f"{name}/dense/kernel"
                snapshot[w_key] = {
                    "weights": module.weight.data.cpu().numpy().copy(),
                    "gradients": _grad_or_zeros(module.weight),
                }
                # Bias
                if module.bias is not None:
                    b_key = f"{name}/dense/bias"
                    snapshot[b_key] = {
                        "weights": module.bias.data.cpu().numpy().copy(),
                        "gradients": _grad_or_zeros(module.bias),
                    }

        self._snapshots[self._batch_number] = snapshot
