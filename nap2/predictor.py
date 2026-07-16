"""NAP2Predictor: high-level orchestrator for the full NAP2 pipeline.

Ties together snapshot collection, statistics extraction, feature map creation,
autoencoder encoding, and LSTM prediction into a single interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.feature_maps import create_feature_map_sequence, log_normalize
from nap2.lstm_predictor import LSTMPredictor
from nap2.snapshot_collector import SnapshotCollector
from nap2.stats import extract_all_stats

DEFAULT_TRAINING_CONFIG = {
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 0.0005,
    "snapshot_interval": 100,
}

# Maps the AE JSON's ``mode`` field to a ``normalize`` value. The AEs shipped
# under controlled_aes/log_transform/ record {"mode": "log_transform"} and
# carry no ``normalize`` key at all.
_AE_MODE_TO_NORMALIZE = {
    "log_transform": "log",
    "log": "log",
    "raw": "none",
    "none": "none",
}


def resolve_normalize(ae_params: Optional[Dict] = None,
                      predictor_params: Optional[Dict] = None) -> tuple:
    """Determine the feature-map normalization the AEs were trained with.

    The transform applied at inference MUST match the one used to train the
    autoencoders: they were fitted on log-transformed maps, so feeding raw
    maps (whose stats span orders of magnitude) puts the AE far outside its
    training distribution and the predictions collapse to the BiGRU's prior.

    The needed value is recorded inconsistently across checkpoint sets, so
    check every known location rather than defaulting to "none" -- a wrong
    default silently disables the transform instead of failing loudly:

    1. AE JSON ``normalize``      (explicit; some checkpoint sets)
    2. AE JSON ``mode``           ("log_transform" -> "log"; michael's AEs)
    3. predictor JSON ``normalize`` ("log"; the BiGRU checkpoint)
    4. "none"                     (last resort)

    Returns:
        ``(normalize, source)`` -- the value plus a human-readable origin,
        so callers can log which file/key it came from.
    """
    if ae_params:
        if ae_params.get("normalize"):
            return str(ae_params["normalize"]), "ae_json:normalize"
        mode = ae_params.get("mode")
        if mode:
            mapped = _AE_MODE_TO_NORMALIZE.get(str(mode))
            if mapped:
                return mapped, f"ae_json:mode={mode}"
    if predictor_params and predictor_params.get("normalize"):
        return str(predictor_params["normalize"]), "predictor_json:normalize"
    return "none", "default(no normalize/mode found)"


class NAP2Predictor:
    """High-level orchestrator that ties the full NAP2 pipeline together.

    1. Partially train a model, collecting weight/gradient snapshots
    2. Extract statistics from snapshots
    3. Create feature maps from statistics
    4. Encode feature maps with pre-trained autoencoders
    5. Predict accuracy with pre-trained LSTM

    Args:
        ae_weights: Pre-trained autoencoder for weight feature maps.
        ae_gradients: Pre-trained autoencoder for gradient feature maps.
        lstm: Pre-trained LSTM predictor.
    """

    def __init__(
        self,
        ae_weights: Optional[FeatureMapAutoEncoder] = None,
        ae_gradients: Optional[FeatureMapAutoEncoder] = None,
        lstm: Optional[nn.Module] = None,
        normalize: str = "none",
    ) -> None:
        self._ae_weights = ae_weights if ae_weights is not None else FeatureMapAutoEncoder()
        self._ae_gradients = ae_gradients if ae_gradients is not None else FeatureMapAutoEncoder()
        self._lstm = lstm if lstm is not None else LSTMPredictor()
        self._normalize = normalize

        self._ae_weights.eval()
        self._ae_gradients.eval()
        self._lstm.eval()

    @classmethod
    def load(cls, model_dir: str) -> "NAP2Predictor":
        """Load from a directory with pre-trained models.

        Expected structure::

            model_dir/ae/weights/last_ae_model.pt
            model_dir/ae/weights/model_hyper_params.json
            model_dir/ae/gradients/last_ae_model.pt
            model_dir/ae/gradients/model_hyper_params.json
            model_dir/lstm/cp/model_state_cp/model.pt
            model_dir/lstm/model_hyper_params.json

        Auto-detects predictor type (lstm or bigru) and normalization mode
        from saved params.
        """
        import json

        base = Path(model_dir)

        # Support both naming conventions: best_ae_model.pt (new) and last_ae_model.pt (legacy)
        def _find_ae_model(ae_dir: Path) -> str:
            for name in ("best_ae_model.pt", "last_ae_model.pt"):
                p = ae_dir / name
                if p.exists():
                    return str(p)
            raise FileNotFoundError(
                f"No AE model found in {ae_dir}. "
                f"Expected best_ae_model.pt or last_ae_model.pt"
            )

        ae_weights = FeatureMapAutoEncoder.load(
            model_path=_find_ae_model(base / "ae" / "weights"),
            params_path=str(base / "ae" / "weights" / "model_hyper_params.json"),
        )
        ae_gradients = FeatureMapAutoEncoder.load(
            model_path=_find_ae_model(base / "ae" / "gradients"),
            params_path=str(base / "ae" / "gradients" / "model_hyper_params.json"),
        )

        # Detect predictor type from LSTM/predictor params
        predictor_params_path = base / "lstm" / "model_hyper_params.json"
        with open(predictor_params_path) as f:
            pred_params = json.load(f)
        predictor_type = pred_params.get("predictor_type", "lstm")

        # Detect normalization. Checks the AE json's `normalize` AND `mode`,
        # falling back to the predictor json -- see resolve_normalize().
        ae_params_path = base / "ae" / "weights" / "model_hyper_params.json"
        with open(ae_params_path) as f:
            ae_params = json.load(f)
        normalize, source = resolve_normalize(ae_params, pred_params)
        logging.info("nap2: feature-map normalize=%s (from %s)", normalize, source)

        model_path = str(base / "lstm" / "cp" / "model_state_cp" / "model.pt")
        params_path = str(predictor_params_path)

        if predictor_type == "bigru":
            predictor = BiGRUDualPredictor.load(
                model_path=model_path, params_path=params_path,
            )
        else:
            predictor = LSTMPredictor.load(
                model_path=model_path, params_path=params_path,
            )

        return cls(ae_weights=ae_weights, ae_gradients=ae_gradients,
                    lstm=predictor, normalize=normalize)

    def score(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        steps: int = 5,
        training_config: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ) -> float:
        """Score a single architecture. Full pipeline in-memory.

        Args:
            model: PyTorch model to evaluate.
            dataloader: Training data loader.
            steps: Number of snapshots to collect.
            training_config: Override default training hyperparameters.
            max_steps: If set, zero-pad the embedding sequence to this length
                before prediction (matches the padded length the predictor was
                trained on; see get_embeddings).

        Returns:
            Predicted accuracy as a float in [0, 1].
        """
        embeddings = self.get_embeddings(model, dataloader, steps, training_config,
                                         max_steps=max_steps)
        return self._lstm.predict(embeddings)

    def score_batch(
        self,
        models: Dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        steps: int = 5,
        training_config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Score multiple architectures.

        Args:
            models: Dictionary mapping model IDs to PyTorch models.
            dataloader: Training data loader.
            steps: Number of snapshots to collect.
            training_config: Override default training hyperparameters.

        Returns:
            Dictionary mapping model IDs to predicted accuracy scores.
        """
        return {
            model_id: self.score(model, dataloader, steps, training_config)
            for model_id, model in models.items()
        }

    def get_embeddings(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        steps: int = 5,
        training_config: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Return AE embeddings for novelty scoring.

        Args:
            model: PyTorch model to evaluate.
            dataloader: Training data loader.
            steps: Number of snapshots to collect.
            training_config: Override default training hyperparameters.
            max_steps: If set and greater than the collected sequence length,
                zero-pad the embedding sequence (real steps first, zeros after)
                up to ``max_steps``. train_lstm.py pads every training sequence
                to max_seq_len and trains on truncated-then-zero-padded copies,
                and predict_anytime.py mirrors this at inference; a raw
                ``[steps, D]`` sequence is otherwise out of distribution for the
                GRU's last-hidden-state path.

        Returns:
            Tensor of shape [max(steps, max_steps), 256] with concatenated
            weight+gradient embeddings.
        """
        config = dict(DEFAULT_TRAINING_CONFIG)
        if training_config is not None:
            config.update(training_config)

        interval = config["snapshot_interval"]

        # Collect snapshots via partial training
        collector = SnapshotCollector(
            model, interval=interval, max_snapshots=steps,
        )
        self._partial_train(model, dataloader, steps, interval, collector, config)

        # Extract statistics
        weight_stats = extract_all_stats(collector.get_weight_snapshots())
        gradient_stats = extract_all_stats(collector.get_gradient_snapshots())

        # Create feature maps: [steps, 65, 100, 12]
        weight_maps = create_feature_map_sequence(weight_stats)
        gradient_maps = create_feature_map_sequence(gradient_stats)

        # Encode feature maps
        embeddings = self._encode_maps(weight_maps, gradient_maps)

        # Zero-pad to the length the predictor was trained on (see docstring).
        if max_steps is not None and max_steps > embeddings.shape[0]:
            pad = torch.zeros(
                max_steps - embeddings.shape[0], embeddings.shape[1],
                dtype=embeddings.dtype,
            )
            embeddings = torch.cat([embeddings, pad], dim=0)

        return embeddings

    def _partial_train(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        steps: int,
        interval: int,
        collector: SnapshotCollector,
        config: Dict,
    ) -> None:
        """Partially train a model, collecting snapshots at regular intervals."""
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
            weight_decay=config["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()

        total_batches = steps * interval
        batch_count = 0
        data_iter = iter(dataloader)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        while batch_count < total_batches:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(model_device, non_blocking=True)
            targets = targets.to(model_device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            # NAS-Bench-201 models return (features, logits) tuple
            if isinstance(outputs, tuple):
                outputs = outputs[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            collector.step()
            batch_count += 1

    @torch.no_grad()
    def _encode_maps(
        self,
        weight_maps: np.ndarray,
        gradient_maps: np.ndarray,
    ) -> torch.Tensor:
        """Encode feature maps with autoencoders and concatenate embeddings.

        Args:
            weight_maps: Array of shape [steps, 65, 100, 12].
            gradient_maps: Array of shape [steps, 65, 100, 12].

        Returns:
            Tensor of shape [steps, 256].
        """
        num_steps = weight_maps.shape[0]
        embeddings = []

        for i in range(num_steps):
            # Reorder: [65, 100, 12] -> [12, 65, 100]
            w_map = np.moveaxis(weight_maps[i], -1, 0)
            g_map = np.moveaxis(gradient_maps[i], -1, 0)

            # Handle NaN -> 0, Inf -> 1e7 (AE training convention)
            w_map = np.where(np.isnan(w_map), 0.0, w_map)
            w_map = np.where(np.isinf(w_map), 1e7, w_map)
            g_map = np.where(np.isnan(g_map), 0.0, g_map)
            g_map = np.where(np.isinf(g_map), 1e7, g_map)

            # Apply feature map normalization (must match AE training)
            if self._normalize == "log":
                w_map = log_normalize(w_map)
                g_map = log_normalize(g_map)

            # Convert to tensors: add batch dim [1, 12, 65, 100]
            w_tensor = torch.tensor(w_map, dtype=torch.float64).unsqueeze(0)
            g_tensor = torch.tensor(g_map, dtype=torch.float64).unsqueeze(0)

            # Encode
            w_emb = self._ae_weights.encode(w_tensor)  # [1, 128]
            g_emb = self._ae_gradients.encode(g_tensor)  # [1, 128]

            # Concatenate: [1, 256]
            combined = torch.cat([w_emb, g_emb], dim=1)
            embeddings.append(combined.squeeze(0))

        return torch.stack(embeddings, dim=0)  # [steps, 256]
