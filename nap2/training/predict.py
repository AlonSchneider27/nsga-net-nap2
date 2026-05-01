"""Generate predictions from trained AE + LSTM pipeline.

Loads trained AE and LSTM, encodes test-set feature maps, runs LSTM inference,
outputs predictions.json: {arch_id: predicted_score}.

Usage:
    python -m nap2.training.predict \
        --maps-dir maps/ \
        --ae-weights-dir models/ae/weights \
        --ae-gradients-dir models/ae/gradients \
        --lstm-dir models/lstm \
        --split-json data/split.json \
        --output predictions.json \
        [--sharpness-dir snapshots/ --sharpness-features F1,F2,F3,F4]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.feature_maps import log_normalize
from nap2.lstm_predictor import LSTMPredictor


def main():
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--maps-dir", type=str, required=True)
    parser.add_argument("--ae-weights-dir", type=str, required=True)
    parser.add_argument("--ae-gradients-dir", type=str, required=True)
    parser.add_argument("--lstm-dir", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True,
                        help="JSON with 'test' key listing test arch IDs")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sharpness-dir", type=str, default=None)
    parser.add_argument("--sharpness-features", type=str, default=None)
    parser.add_argument("--sharpness-only", action="store_true")
    args = parser.parse_args()

    # Load split
    with open(args.split_json) as f:
        split = json.load(f)
    test_ids = set(split["test"])

    # Load models
    ae_weights = None
    ae_gradients = None
    if not args.sharpness_only:
        ae_weights = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_weights_dir) / "ae_weights.pt"),
            params_path=str(Path(args.ae_weights_dir) / "aew_model_hyper_params.json"),
        )
        ae_gradients = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_gradients_dir) / "ae_gradients.pt"),
            params_path=str(Path(args.ae_gradients_dir) / "model_hyper_params.json"),
        )

    # Load predictor (auto-detect type)
    lstm_params_path = Path(args.lstm_dir) / "lstm_model_hyper_params.json"
    with open(lstm_params_path) as f:
        lstm_params = json.load(f)

    predictor_type = lstm_params.get("predictor_type", "lstm")
    lstm_model_path = Path(args.lstm_dir) / "cp" / "model_state_cp" / "lstm_reg_final.pt"

    if predictor_type == "bigru":
        lstm = BiGRUDualPredictor.load(
            model_path=str(lstm_model_path), params_path=str(lstm_params_path),
        )
    else:
        lstm = LSTMPredictor.load(
            model_path=str(lstm_model_path), params_path=str(lstm_params_path),
        )

    # Detect normalization from AE params
    ae_normalize = "none"
    if not args.sharpness_only:
        ae_params_path = Path(args.ae_weights_dir) / "model_hyper_params.json"
        if ae_params_path.exists():
            with open(ae_params_path) as f:
                ae_params = json.load(f)
            ae_normalize = ae_params.get("normalize", "none")

    # Load sharpness if provided
    sharpness_data = {}
    sharpness_feature_names = []
    if args.sharpness_dir:
        sharpness_feature_names = (
            args.sharpness_features.split(",") if args.sharpness_features
            else ["F1", "F2", "F3", "F4"]
        )
        for sf_path in Path(args.sharpness_dir).glob("*_sharpness.pkl"):
            aid = sf_path.stem.replace("_sharpness", "")
            with open(str(sf_path), "rb") as f:
                sharpness_data[aid] = pickle.load(f)

    # Generate predictions for test set
    maps_dir = Path(args.maps_dir)
    predictions = {}

    if args.sharpness_only:
        for arch_id in test_ids:
            if arch_id not in sharpness_data:
                continue
            sf = sharpness_data[arch_id]
            num_steps = len(sf["F1"])
            step_embeddings = []
            for i in range(num_steps):
                vec = torch.tensor(
                    [sf[feat][i] for feat in sharpness_feature_names],
                    dtype=torch.float64,
                )
                step_embeddings.append(vec)
            embedding_seq = torch.stack(step_embeddings, dim=0).unsqueeze(0)
            with torch.no_grad():
                score = lstm(embedding_seq).item()
            predictions[arch_id] = score
    else:
        for wf in sorted(maps_dir.glob("*_weights_maps.pkl")):
            arch_id = wf.stem.replace("_weights_maps", "")
            if arch_id not in test_ids:
                continue

            gf = maps_dir / f"{arch_id}_gradients_maps.pkl"
            if not gf.exists():
                continue

            with open(str(wf), "rb") as f:
                weight_maps = pickle.load(f)
            with open(str(gf), "rb") as f:
                gradient_maps = pickle.load(f)

            num_steps = weight_maps.shape[0]
            step_embeddings = []

            for i in range(num_steps):
                w_map = np.moveaxis(weight_maps[i], -1, 0)
                g_map = np.moveaxis(gradient_maps[i], -1, 0)
                w_map = np.where(np.isnan(w_map), 0.0, w_map)
                w_map = np.where(np.isinf(w_map), 1e7, w_map)
                g_map = np.where(np.isnan(g_map), 0.0, g_map)
                g_map = np.where(np.isinf(g_map), 1e7, g_map)

                if ae_normalize == "log":
                    w_map = log_normalize(w_map)
                    g_map = log_normalize(g_map)

                w_tensor = torch.tensor(w_map, dtype=torch.float64).unsqueeze(0)
                g_tensor = torch.tensor(g_map, dtype=torch.float64).unsqueeze(0)

                with torch.no_grad():
                    w_emb = ae_weights.encode(w_tensor)
                    g_emb = ae_gradients.encode(g_tensor)
                combined = torch.cat([w_emb, g_emb], dim=1).squeeze(0)

                if sharpness_feature_names and arch_id in sharpness_data:
                    sf = sharpness_data[arch_id]
                    sharpness_vec = torch.tensor(
                        [sf[feat][i] for feat in sharpness_feature_names],
                        dtype=torch.float64,
                    )
                    combined = torch.cat([combined, sharpness_vec])

                step_embeddings.append(combined)

            embedding_seq = torch.stack(step_embeddings, dim=0).unsqueeze(0)  # [1, seq, dim]
            with torch.no_grad():
                score = lstm(embedding_seq).item()
            predictions[arch_id] = score

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Generated predictions for {len(predictions)} test architectures")


if __name__ == "__main__":
    main()
