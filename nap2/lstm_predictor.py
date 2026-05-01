"""LSTM-based performance predictor.

Takes a sequence of autoencoder embeddings (weights + gradients = 256-dim per
step) and predicts the final accuracy of the network (0.0 - 1.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM -> Dense -> Sigmoid predictor for neural architecture performance.

    Parameters
    ----------
    embedding_size : int
        Dimensionality of each input timestep (default 256 = 128 weights + 128 gradients).
    hidden_size : int
        LSTM hidden state size.
    lstm_layers : int
        Number of stacked LSTM layers.
    inner_dense_sizes : tuple of int
        Sizes of intermediate dense layers between LSTM output and final Linear(*, 1).
    last_layer : str
        Activation after the final Linear layer: ``"sigmoid"``, ``"relu"``, or ``"linear"``.
    bi_directional : bool
        Whether the LSTM is bidirectional.
    batch_first : bool
        If ``True`` input is ``[B, seq, features]``.
    is_double : bool
        If ``True`` use float64 precision.
    """

    def __init__(
        self,
        embedding_size: int = 256,
        hidden_size: int = 2048,
        lstm_layers: int = 1,
        inner_dense_sizes: Sequence[int] = (64,),
        last_layer: str = "sigmoid",
        bi_directional: bool = False,
        batch_first: bool = True,
        is_double: bool = True,
    ) -> None:
        super().__init__()
        self._is_double = is_double

        # LSTM
        self._lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=batch_first,
            bidirectional=bi_directional,
        )

        # Dense head
        dense_input_size = hidden_size * 2 if bi_directional else hidden_size
        layers: list[nn.Module] = []

        prev_size = dense_input_size
        for size in inner_dense_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        # Final projection to scalar
        layers.append(nn.Linear(prev_size, 1))
        if last_layer == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_layer == "relu":
            layers.append(nn.ReLU())
        elif last_layer != "linear":
            raise ValueError(f"Unsupported last_layer: {last_layer!r}")

        self._dense_layers = nn.ModuleList(layers)

        if self._is_double:
            self.double()

    # ── Forward / predict ────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``[B, seq, embedding_size] -> [B, 1]``."""
        lstm_out, _ = self._lstm(x)
        x = lstm_out[:, -1, :]  # last timestep
        for layer in self._dense_layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def predict(self, embedding_sequence: torch.Tensor) -> float:
        """Predict accuracy from a single embedding sequence.

        Parameters
        ----------
        embedding_sequence : Tensor
            Shape ``[seq, embedding_size]`` (single sample, no batch dim).

        Returns
        -------
        float
            Predicted accuracy in [0, 1].
        """
        self.eval()
        x = embedding_sequence.unsqueeze(0)  # add batch dim
        out = self.forward(x)
        return out.item()

    # ── Persistence ──────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        params_path: Optional[Union[str, Path]] = None,
    ) -> "LSTMPredictor":
        """Load a saved LSTM predictor.

        Parameters
        ----------
        model_path : path-like
            Path to the ``state_dict`` (``.pt`` file).
        params_path : path-like, optional
            Path to a JSON file with constructor kwargs.  If ``None`` the file
            ``lstm_params.json`` next to *model_path* is tried.
        """
        model_path = Path(model_path)

        if params_path is None:
            params_path = model_path.parent / "lstm_params.json"
        else:
            params_path = Path(params_path)

        with open(params_path) as f:
            raw = json.load(f)

        # Map stored param names to constructor kwarg names
        kwargs = {
            "embedding_size": raw["embedding_size"],
            "hidden_size": raw["hidden_size"],
            "lstm_layers": raw["lstm_layers"],
            "inner_dense_sizes": tuple(raw["inner_dense_layer_sizes"]),
            "last_layer": raw["last_layer"],
            "bi_directional": raw["bi_directional"],
            "batch_first": raw["batch_first"],
            "is_double": raw["is_double"],
        }

        model = cls(**kwargs)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
