"""Convolutional autoencoder for compressing feature maps to embeddings."""

import json
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU


DEFAULT_LAYER_SHAPES = [[[1, 100], 512], [[65, 1], 256], [[1, 1], 128]]


class FeatureMapAutoEncoder(Module):
    """Convolutional autoencoder that compresses feature maps to fixed-size embeddings.

    Compresses input tensors of shape [B, 12, 65, 100] (stats x layers x values)
    down to a 1D embedding vector. Separate instances are trained for weight
    statistics and gradient statistics.
    """

    def __init__(
        self,
        layer_shapes: Optional[List] = None,
        input_channels: int = 12,
    ):
        super().__init__()
        if layer_shapes is None:
            layer_shapes = DEFAULT_LAYER_SHAPES
        self._layer_shapes = layer_shapes
        self._input_channels = input_channels

        # Build encoder: Conv2d + BatchNorm2d + ReLU per layer
        curr_in_channels = input_channels
        self._encoder = Sequential()
        for kernel_size, out_channels in layer_shapes:
            self._encoder.append(
                Conv2d(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            curr_in_channels = out_channels
            self._encoder.append(BatchNorm2d(curr_in_channels))
            self._encoder.append(ReLU())

        # Build decoder: ConvTranspose2d + BatchNorm2d + ReLU per reversed layer,
        # plus a final ConvTranspose2d(kernel=1x1) to reconstruct input channels
        self._decoder = Sequential()
        for kernel_size, out_channels in reversed(layer_shapes):
            self._decoder.append(
                ConvTranspose2d(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            curr_in_channels = out_channels
            self._decoder.append(BatchNorm2d(curr_in_channels))
            self._decoder.append(ReLU())
        self._decoder.append(
            ConvTranspose2d(
                in_channels=curr_in_channels,
                out_channels=input_channels,
                kernel_size=(1, 1),
            )
        )

        # Use float64 to match original training
        self.double()

    def encode(self, x: Tensor) -> Tensor:
        """Encode input feature maps to flat embedding vectors.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Flattened embedding of shape [B, embed_dim].
        """
        h = self._encoder(x)
        return h.flatten(start_dim=1)

    def decode(self, x: Tensor) -> Tensor:
        """Decode flat embedding vectors back to feature map shape.

        Args:
            x: Flattened embedding of shape [B, embed_dim].

        Returns:
            Reconstructed tensor of shape [B, C, H, W].
        """
        # Infer spatial dims from the encoder output shape
        # For default config: encoder output is [B, 128, 1, 1]
        # so embed_dim = 128 * 1 * 1 = 128
        # We need to reshape back to [B, C_last, H_last, W_last]
        last_channels = self._layer_shapes[-1][1]
        spatial = x.shape[1] // last_channels
        # Assuming spatial is h*w where both are 1 for default config
        # For generality, compute from layer_shapes
        h = x.view(x.shape[0], last_channels, 1, -1)
        return self._decoder(h)

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Reconstructed tensor of same shape as input.
        """
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        return decoded

    @classmethod
    def load(
        cls,
        model_path: str,
        params_path: Optional[str] = None,
    ) -> "FeatureMapAutoEncoder":
        """Load a pre-trained autoencoder from disk.

        Args:
            model_path: Path to the saved state dict (.pt file).
            params_path: Path to the JSON params file. If None, uses defaults.

        Returns:
            Loaded model in eval mode.
        """
        if params_path is not None:
            with open(params_path) as f:
                params = json.load(f)
            layer_shapes = params["layers_shapes"]
        else:
            layer_shapes = None

        input_channels = params.get("input_channels", 12) if params_path else 12
        model = cls(layer_shapes=layer_shapes, input_channels=input_channels)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
