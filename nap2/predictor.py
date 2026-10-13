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


# Order in which the two 128-dim AE embeddings are concatenated into the
# predictor's 256-dim input. It is a property of the predictor checkpoint:
#   "wg"  [weights, gradients] -- the NAPv2 repo's order (train_lstm.py,
#         predict.py, NAP2Predictor), i.e. every model trained with the repo
#         (e.g. NAPv2_assets/ae_lognorm_seed42/*). The default.
#   "gw"  [gradients, weights] -- michael's deployed checkpoints only
#         (cross-dataset-autoresearch/lookup_tables/*/model.pt, trained by
#         older code). Must be declared in their predictor JSON.
# The wrong order raises no error: predictions collapse to a near-constant
# value and KT falls to ~0.3, so the resolved value is logged at load time.
EMBEDDING_ORDERS = ("wg", "gw")


def resolve_snapshot_interval(predictor_params: Optional[Dict] = None) -> tuple:
    """Mini-batches between snapshots the predictor was trained on.

    Reads ``snapshot_interval`` from the predictor JSON; absent means the
    historical 100 (michael's checkpoints). Returns ``(interval, source)``.
    """
    if predictor_params and predictor_params.get("snapshot_interval") is not None:
        iv = int(predictor_params["snapshot_interval"])
        if iv < 1:
            raise ValueError(f"predictor JSON snapshot_interval={iv!r}; expected a positive int")
        return iv, "predictor_json:snapshot_interval"
    return int(DEFAULT_TRAINING_CONFIG["snapshot_interval"]), "default(100)"


def resolve_embedding_order(predictor_params: Optional[Dict] = None) -> tuple:
    """Return ``(order, source)`` for the AE-embedding concat order.

    Reads ``embedding_order`` from the predictor JSON; absent means "wg".
    Any other value raises rather than guessing.
    """
    if predictor_params and predictor_params.get("embedding_order") is not None:
        order = str(predictor_params["embedding_order"])
        if order not in EMBEDDING_ORDERS:
            raise ValueError(
                f"predictor JSON embedding_order={order!r}; expected one of {EMBEDDING_ORDERS}")
        return order, "predictor_json:embedding_order"
    return "wg", "default([weights, gradients], the NAPv2 repo order)"


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
        normalize: Feature-map normalization the AEs were trained with.
        embedding_order: "wg" (default) or "gw", see EMBEDDING_ORDERS.
    """

    def __init__(
        self,
        ae_weights: Optional[FeatureMapAutoEncoder] = None,
        ae_gradients: Optional[FeatureMapAutoEncoder] = None,
        lstm: Optional[nn.Module] = None,
        normalize: str = "none",
        embedding_order: str = "wg",
        snapshot_interval: int = DEFAULT_TRAINING_CONFIG["snapshot_interval"],
    ) -> None:
        if embedding_order not in EMBEDDING_ORDERS:
            raise ValueError(
                f"embedding_order={embedding_order!r}; expected one of {EMBEDDING_ORDERS}")
        self._ae_weights = ae_weights if ae_weights is not None else FeatureMapAutoEncoder()
        self._ae_gradients = ae_gradients if ae_gradients is not None else FeatureMapAutoEncoder()
        self._lstm = lstm if lstm is not None else LSTMPredictor()
        self._normalize = normalize
        self._embedding_order = embedding_order
        # Cadence the predictor was trained on; used as the partial-training
        # snapshot interval unless training_config overrides it explicitly.
        self.snapshot_interval = int(snapshot_interval)

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
        embedding_order, order_src = resolve_embedding_order(pred_params)
        logging.info("nap2: embedding order=%s (from %s)", embedding_order, order_src)
        snapshot_interval, iv_src = resolve_snapshot_interval(pred_params)
        logging.info("nap2: snapshot interval=%d mb (from %s)", snapshot_interval, iv_src)

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
                    lstm=predictor, normalize=normalize,
                    embedding_order=embedding_order,
                    snapshot_interval=snapshot_interval)

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
        weight_maps, gradient_maps = self.collect_maps(
            model, dataloader, steps, training_config)
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

    def collect_maps(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        steps: int = 5,
        training_config: Optional[Dict] = None,
    ):
        """Partially train ``model`` and return the raw feature-map sequences.

        The snapshot interval defaults to ``self.snapshot_interval`` (the
        cadence this predictor was trained on); ``training_config`` may
        override it. Returns ``(weight_maps, gradient_maps)``, each of shape
        [steps, 65, 100, 12], un-normalized -- the AE-specific normalization
        is applied in ``_encode_maps``, so one collection can be encoded by
        several predictors (different AEs) of the same cadence.
        """
        config = dict(DEFAULT_TRAINING_CONFIG)
        config["snapshot_interval"] = self.snapshot_interval
        if training_config is not None:
            config.update(training_config)
        interval = int(config["snapshot_interval"])

        collector = SnapshotCollector(
            model, interval=interval, max_snapshots=steps,
        )
        self._partial_train(model, dataloader, steps, interval, collector, config)

        weight_stats = extract_all_stats(collector.get_weight_snapshots())
        gradient_stats = extract_all_stats(collector.get_gradient_snapshots())
        return (create_feature_map_sequence(weight_stats),
                create_feature_map_sequence(gradient_stats))

    def encode_maps(self, weight_maps: np.ndarray, gradient_maps: np.ndarray) -> torch.Tensor:
        """Public alias of ``_encode_maps`` (normalize + AE encode + concat)."""
        return self._encode_maps(weight_maps, gradient_maps)

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

            # MPS executes asynchronously; without a sync the collector's
            # .cpu().numpy() copies can read unfinished buffers (observed:
            # entire conv layers captured as NaN while live training is
            # healthy). CUDA/CPU are unaffected.
            if model_device.type == "mps":
                torch.mps.synchronize()
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

            # Concatenate: [1, 256] in the checkpoint's order (EMBEDDING_ORDERS).
            # "gw" exists only for michael's deployed checkpoints: verified
            # against his embedding cache (cifar10_via_log_cifar10.pkl), their
            # dims 0-127 are the gradient embedding. Everything trained with
            # the NAPv2 repo is "wg". The wrong order collapses predictions to
            # a near-constant value without any error.
            if self._embedding_order == "gw":
                combined = torch.cat([g_emb, w_emb], dim=1)
            else:
                combined = torch.cat([w_emb, g_emb], dim=1)
            embeddings.append(combined.squeeze(0))

        return torch.stack(embeddings, dim=0)  # [steps, 256]
