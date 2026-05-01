"""Bidirectional GRU dual-path performance predictor.

Takes a sequence of autoencoder embeddings (weights + gradients = 256-dim per
step) and predicts the final accuracy of the network (0.0 - 1.0).

Dual-path readout: last hidden state (final-timestep summary) concatenated
with attention-pooled context (global temporal context).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class BiGRUDualPredictor(nn.Module):
    """Bidirectional GRU with dual pathway: last hidden + attention pooling.

    Parameters
    ----------
    embedding_size : int
        Dimensionality of each input timestep (default 256 = 128 weights + 128 gradients).
    hidden_size : int
        GRU hidden state size per direction.
    gru_layers : int
        Number of stacked GRU layers.
    dense_size : int
        Size of the dense layer between dual-path output and final Linear(*, 1).
    dropout : float
        Dropout probability applied to GRU inter-layer and after dual-path concat.
    last_layer : str
        Activation after the final Linear layer: ``"sigmoid"``, ``"relu"``, or ``"linear"``.
    is_double : bool
        If ``True`` use float64 precision.
    """

    def __init__(
        self,
        embedding_size: int = 256,
        hidden_size: int = 128,
        gru_layers: int = 2,
        dense_size: int = 128,
        dropout: float = 0.1,
        last_layer: str = "sigmoid",
        is_double: bool = True,
    ) -> None:
        super().__init__()
        self._is_double = is_double

        self._gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        bidir_size = hidden_size * 2  # forward + backward

        # Attention pathway
        self._attn_w = nn.Linear(bidir_size, 1)

        # Dense head: last_hidden (bidir_size) + attn_context (bidir_size) = bidir_size * 2
        self._dropout = nn.Dropout(dropout)
        self._dense = nn.Linear(bidir_size * 2, dense_size)
        self._relu = nn.ReLU()
        self._out = nn.Linear(dense_size, 1)

        if last_layer == "sigmoid":
            self._last_act = nn.Sigmoid()
        elif last_layer == "relu":
            self._last_act = nn.ReLU()
        elif last_layer == "linear":
            self._last_act = nn.Identity()
        else:
            raise ValueError(f"Unsupported last_layer: {last_layer!r}")

        if self._is_double:
            self.double()

    # -- Forward / predict ------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ``[B, seq, embedding_size] -> [B, 1]``."""
        out, h = self._gru(x)  # out: [B, T, hidden*2], h: [layers*2, B, hidden]

        # Path 1: last hidden state (concat fwd/bwd of last layer)
        last_h = torch.cat([h[-2], h[-1]], dim=-1)  # [B, hidden*2]

        # Path 2: attention pooling over all timesteps
        attn_scores = self._attn_w(out).squeeze(-1)  # [B, T]
        # Mask zero-padded steps
        mask = (x.abs().sum(dim=-1) > 0).float()  # [B, T]
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        context = (out * attn_weights).sum(dim=1)  # [B, hidden*2]

        # Combine both paths
        combined = torch.cat([last_h, context], dim=-1)  # [B, hidden*4]
        combined = self._dropout(combined)
        h_out = self._relu(self._dense(combined))
        return self._last_act(self._out(h_out))

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

    # -- Persistence -------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        params_path: Optional[Union[str, Path]] = None,
    ) -> "BiGRUDualPredictor":
        """Load a saved BiGRU predictor.

        Parameters
        ----------
        model_path : path-like
            Path to the ``state_dict`` (``.pt`` file).
        params_path : path-like, optional
            Path to a JSON file with constructor kwargs.  If ``None`` the file
            ``predictor_params.json`` next to *model_path* is tried.
        """
        model_path = Path(model_path)

        if params_path is None:
            # Try both naming conventions
            params_path = model_path.parent / "predictor_params.json"
            if not params_path.exists():
                params_path = model_path.parent / "lstm_params.json"
        else:
            params_path = Path(params_path)

        with open(params_path) as f:
            raw = json.load(f)

        kwargs = {
            "embedding_size": raw["embedding_size"],
            "hidden_size": raw.get("hidden_size", 128),
            "gru_layers": raw.get("gru_layers", raw.get("lstm_layers", 2)),
            "dense_size": raw.get("dense_size", raw.get("inner_dense_layer_sizes", [128])[0]),
            "dropout": raw.get("dropout", 0.1),
            "last_layer": raw.get("last_layer", "sigmoid"),
            "is_double": raw.get("is_double", True),
        }

        model = cls(**kwargs)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
