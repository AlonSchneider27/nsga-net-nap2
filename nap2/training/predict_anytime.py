"""Anytime prediction evaluation — validates augmentation training.

For each truncation step t (1, 2, ..., max_steps), predicts architecture
performance using only the first t embedding steps (zero-padding the rest).
Computes Kendall Tau at each step to verify that prediction quality
increases with more training steps observed.

This validates that --aug training enables the LSTM to predict accurately
from any number of steps, not just the full sequence.

Usage:
    python -m nap2.training.predict_anytime \
        --maps-dir maps/ \
        --ae-weights-dir models/ae/weights \
        --ae-gradients-dir models/ae/gradients \
        --lstm-dir models/lstm \
        --split-json data/fold1_split.json \
        --results-json data/results.json \
        --eval-set val \
        --output anytime_kt.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.feature_maps import log_normalize
from nap2.lstm_predictor import LSTMPredictor


def _encode_single_arch(arch_id, maps_dir, ae_weights, ae_gradients,
                        sharpness_data=None, sharpness_feature_names=None,
                        normalize="none"):
    """Encode one architecture's feature maps into an embedding sequence."""
    wf = maps_dir / f"{arch_id}_weights_maps.pkl"
    gf = maps_dir / f"{arch_id}_gradients_maps.pkl"

    if not wf.exists() or not gf.exists():
        return None

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

        if normalize == "log":
            w_map = log_normalize(w_map)
            g_map = log_normalize(g_map)

        w_tensor = torch.tensor(w_map, dtype=torch.float64).unsqueeze(0)
        g_tensor = torch.tensor(g_map, dtype=torch.float64).unsqueeze(0)

        with torch.no_grad():
            w_emb = ae_weights.encode(w_tensor)
            g_emb = ae_gradients.encode(g_tensor)
        combined = torch.cat([w_emb, g_emb], dim=1).squeeze(0)

        if sharpness_feature_names and sharpness_data and arch_id in sharpness_data:
            sf = sharpness_data[arch_id]
            sharpness_vec = torch.tensor(
                [sf[feat][i] for feat in sharpness_feature_names],
                dtype=torch.float64,
            )
            combined = torch.cat([combined, sharpness_vec])

        step_embeddings.append(combined)

    return torch.stack(step_embeddings, dim=0)  # [num_steps, embed_dim]


def main():
    parser = argparse.ArgumentParser(
        description="Anytime prediction evaluation (KT at each truncation step)")
    parser.add_argument("--maps-dir", type=str, required=True)
    parser.add_argument("--ae-weights-dir", type=str, required=True)
    parser.add_argument("--ae-gradients-dir", type=str, required=True)
    parser.add_argument("--lstm-dir", type=str, required=True)
    parser.add_argument("--split-json", type=str, required=True)
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--eval-set", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to evaluate on (default: val)")
    parser.add_argument("--sharpness-dir", type=str, default=None,
                        help="Directory with sharpness pkl files (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Load split and ground truth
    with open(args.split_json) as f:
        split = json.load(f)
    eval_ids = split[args.eval_set]

    with open(args.results_json) as f:
        results = json.load(f)

    # Load LSTM params first to determine mode
    lstm_params_path = Path(args.lstm_dir) / "model_hyper_params.json"
    with open(lstm_params_path) as f:
        lstm_params = json.load(f)
    predictor_type = lstm_params.get("predictor_type", "lstm")
    lstm_model_path = str(Path(args.lstm_dir) / "cp" / "model_state_cp" / "model.pt")

    if predictor_type == "bigru":
        lstm = BiGRUDualPredictor.load(
            model_path=lstm_model_path, params_path=str(lstm_params_path),
        )
    else:
        lstm = LSTMPredictor.load(
            model_path=lstm_model_path, params_path=str(lstm_params_path),
        )

    # Load AEs (skip if sharpness-only)
    sharpness_only = lstm_params.get("sharpness_only", False)
    ae_weights = None
    ae_gradients = None
    ae_normalize = "none"
    if not sharpness_only:
        ae_weights = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_weights_dir) / "best_ae_model.pt"),
            params_path=str(Path(args.ae_weights_dir) / "model_hyper_params.json"),
        )
        ae_gradients = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_gradients_dir) / "best_ae_model.pt"),
            params_path=str(Path(args.ae_gradients_dir) / "model_hyper_params.json"),
        )
        # Detect normalization from AE params
        ae_params_path = Path(args.ae_weights_dir) / "model_hyper_params.json"
        if ae_params_path.exists():
            with open(ae_params_path) as f:
                ae_params = json.load(f)
            ae_normalize = ae_params.get("normalize", "none")

    # Load sharpness data if the LSTM was trained with it
    sharpness_data = {}
    sharpness_feature_names = lstm_params.get("sharpness_features") or []
    sharpness_norm = lstm_params.get("sharpness_norm")
    if sharpness_feature_names and args.sharpness_dir:
        sharpness_dir = Path(args.sharpness_dir)
        for sf_path in sharpness_dir.glob("*_sharpness.pkl"):
            aid = sf_path.stem.replace("_sharpness", "")
            with open(str(sf_path), "rb") as f:
                sharpness_data[aid] = pickle.load(f)
        print(f"Loaded sharpness data for {len(sharpness_data)} architectures")
        print(f"Using sharpness features: {sharpness_feature_names}")
    elif sharpness_feature_names and not args.sharpness_dir:
        print(f"WARNING: LSTM was trained with sharpness features {sharpness_feature_names} "
              f"but --sharpness-dir not provided. Results will be incorrect.")

    # Encode all eval architectures
    maps_dir = Path(args.maps_dir)
    arch_embeddings = {}  # arch_id -> [num_steps, embed_dim]
    print(f"Encoding {len(eval_ids)} {args.eval_set} architectures...")
    for arch_id in sorted(eval_ids):
        if arch_id not in results:
            continue
        if sharpness_only:
            # Sharpness-only: no AE needed
            if arch_id not in sharpness_data:
                continue
            sf = sharpness_data[arch_id]
            num_steps = len(sf[sharpness_feature_names[0]])
            step_embs = []
            for i in range(num_steps):
                vec = torch.tensor(
                    [sf[feat][i] for feat in sharpness_feature_names],
                    dtype=torch.float64,
                )
                step_embs.append(vec)
            arch_embeddings[arch_id] = torch.stack(step_embs, dim=0)
        else:
            emb = _encode_single_arch(arch_id, maps_dir, ae_weights, ae_gradients,
                                      sharpness_data, sharpness_feature_names,
                                      normalize=ae_normalize)
            if emb is not None:
                arch_embeddings[arch_id] = emb

    # Apply z-score normalization using train stats saved in model params
    if sharpness_norm and sharpness_feature_names:
        mean = torch.tensor(sharpness_norm["mean"], dtype=torch.float64)
        std = torch.tensor(sharpness_norm["std"], dtype=torch.float64)
        n_sharp = len(sharpness_feature_names)
        for arch_id, emb in arch_embeddings.items():
            arch_embeddings[arch_id] = emb.clone()
            arch_embeddings[arch_id][:, -n_sharp:] = (emb[:, -n_sharp:] - mean) / std
        print(f"Applied sharpness normalization from training stats")

    if not arch_embeddings:
        print("No valid architectures found.")
        return

    max_steps = max(e.shape[0] for e in arch_embeddings.values())
    embed_dim = next(iter(arch_embeddings.values())).shape[1]
    print(f"Encoded {len(arch_embeddings)} architectures, "
          f"max_steps={max_steps}, embed_dim={embed_dim}")

    # Ground truth values
    ground_truth = {aid: results[aid] for aid in arch_embeddings}
    true_vals = np.array([ground_truth[aid] for aid in sorted(arch_embeddings)])

    # Evaluate at each truncation step
    print(f"\n{'Step':>4}  {'KT':>7}  {'Spearman':>9}  {'N':>4}")
    print("-" * 32)

    step_results = []
    for t in range(1, max_steps + 1):
        predictions = {}
        for arch_id, emb in arch_embeddings.items():
            # Zero-pad: keep first t steps, zero the rest (matches aug training)
            truncated = emb.clone()
            if t < emb.shape[0]:
                truncated[t:] = 0.0
            seq = truncated.unsqueeze(0)  # [1, num_steps, embed_dim]
            # Alternative: actual truncation (no zero-pad, monotonic KT curve)
            # seq = emb[:t].unsqueeze(0)  # [1, t, embed_dim]
            with torch.no_grad():
                score = lstm(seq).item()
            predictions[arch_id] = score

        # Compute KT
        common = sorted(set(predictions) & set(ground_truth))
        pred_vals = np.array([predictions[k] for k in common])
        gt_vals = np.array([ground_truth[k] for k in common])

        kt, _ = kendalltau(pred_vals, gt_vals)
        from scipy.stats import spearmanr
        sr, _ = spearmanr(pred_vals, gt_vals)

        step_results.append({
            "step": t,
            "kendall_tau": float(kt),
            "spearman_rho": float(sr),
            "n": len(common),
        })

        print(f"{t:>4}  {kt:>7.4f}  {sr:>9.4f}  {len(common):>4}")

    # Summary
    print(f"\nStep  1: KT = {step_results[0]['kendall_tau']:.4f}")
    print(f"Step {max_steps:>2}: KT = {step_results[-1]['kendall_tau']:.4f}")
    kt_values = [r['kendall_tau'] for r in step_results]
    print(f"Monotonic increase: {all(kt_values[i] <= kt_values[i+1] for i in range(len(kt_values)-1))}")
    print(f"Overall trend: {'increasing' if kt_values[-1] > kt_values[0] else 'not increasing'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(step_results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
