"""Train the LSTM performance predictor.

Loads feature maps, encodes them with pre-trained autoencoders, and trains
an LSTMPredictor to predict final accuracy from embedding sequences.
Best model checkpoint is saved based on validation loss.

Requires --split-json with 'train' and 'val' keys listing architecture IDs.
This ensures LSTM uses the same train/val split as the AE (no data leakage).

Optionally concatenates sharpness features (F1-F4) to AE embeddings,
or uses sharpness features alone (--sharpness-only).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.feature_maps import log_normalize
from nap2.lstm_predictor import LSTMPredictor
from nap2.training.warmup import WarmUpLR


def _encode_architectures(arch_ids, maps_dir, ae_weights, ae_gradients,
                          results, sharpness_data, sharpness_feature_names,
                          normalize="none"):
    """Encode feature maps into embedding sequences for a list of arch IDs."""
    embeddings = []
    targets = []
    sorted_ids = sorted(arch_ids)
    n_total = len(sorted_ids)
    skipped = 0

    for idx, arch_id in enumerate(sorted_ids):
        wf = maps_dir / f"{arch_id}_weights_maps.pkl"
        gf = maps_dir / f"{arch_id}_gradients_maps.pkl"

        if not wf.exists() or not gf.exists():
            skipped += 1
            continue
        if arch_id not in results:
            skipped += 1
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

            if normalize == "log":
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

        embedding_seq = torch.stack(step_embeddings, dim=0)
        embeddings.append(embedding_seq)
        targets.append(results[arch_id])

        if (idx + 1) % 100 == 0 or idx + 1 == n_total:
            print(f"  Encoded {idx + 1}/{n_total} architectures "
                  f"({len(embeddings)} valid, {skipped} skipped)", flush=True)

    return embeddings, targets


def _encode_sharpness_only(arch_ids, results, sharpness_data, sharpness_feature_names):
    """Encode architectures using sharpness features only (no AE)."""
    embeddings = []
    targets = []

    for arch_id in sorted(arch_ids):
        if arch_id not in sharpness_data or arch_id not in results:
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
        embedding_seq = torch.stack(step_embeddings, dim=0)
        embeddings.append(embedding_seq)
        targets.append(results[arch_id])

    return embeddings, targets


def _normalize_sharpness(embeddings, n_sharpness, stats=None):
    """Z-score normalize sharpness feature dimensions of embedding sequences.

    Normalizes the last n_sharpness dims of each [seq_len, embed_dim] tensor.
    If stats is None, computes mean/std from the data (use for train set).
    Otherwise applies provided stats (use for val/test set).

    Returns (normalized_embeddings, stats_dict).
    """
    if n_sharpness == 0:
        return embeddings, None

    if stats is None:
        # Collect all sharpness values: [n_archs * seq_len, n_sharpness]
        all_sharp = torch.cat([emb[:, -n_sharpness:] for emb in embeddings], dim=0)
        mean = all_sharp.mean(dim=0)
        std = all_sharp.std(dim=0)
        # Avoid division by zero for constant features
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        stats = {"mean": mean, "std": std}

    normalized = []
    for emb in embeddings:
        emb = emb.clone()
        emb[:, -n_sharpness:] = (emb[:, -n_sharpness:] - stats["mean"]) / stats["std"]
        normalized.append(emb)

    return normalized, stats


def _pad_and_stack(embeddings):
    """Pad embedding sequences to same length and stack into tensor."""
    max_seq_len = max(e.shape[0] for e in embeddings)
    embedding_size = embeddings[0].shape[1]

    padded = []
    for emb in embeddings:
        if emb.shape[0] < max_seq_len:
            padding = torch.zeros(max_seq_len - emb.shape[0], embedding_size,
                                  dtype=torch.float64)
            emb = torch.cat([emb, padding], dim=0)
        padded.append(emb)

    return torch.stack(padded, dim=0), embedding_size


def _augment_sequences(embeddings, targets, aug_steps):
    """Create truncated subsequences for anytime prediction training.

    For each architecture with full sequence of length S, creates aug_steps
    truncated copies (length 1, 2, ..., aug_steps) plus the full sequence.
    Truncated sequences are zero-padded to full length S.
    All copies share the same target (final accuracy).

    This trains the LSTM to predict accurately from any number of
    training steps, not just the full sequence.

    Args:
        embeddings: List of tensors [seq_len, embed_dim], one per architecture.
        targets: List of float targets, one per architecture.
        aug_steps: Number of truncated copies (1..aug_steps). 0 to disable.

    Returns:
        Augmented (embeddings, targets) lists.
    """
    if aug_steps <= 0:
        return embeddings, targets

    aug_embeddings = []
    aug_targets = []

    for emb, tgt in zip(embeddings, targets):
        seq_len = emb.shape[0]
        # Full sequence
        aug_embeddings.append(emb)
        aug_targets.append(tgt)
        # Truncated sequences: keep first k steps, zero-pad the rest
        for k in range(1, min(aug_steps, seq_len) + 1):
            truncated = emb.clone()
            truncated[k:] = 0.0
            aug_embeddings.append(truncated)
            aug_targets.append(tgt)

    return aug_embeddings, aug_targets


def main():
    parser = argparse.ArgumentParser(description="Train LSTM predictor")
    parser.add_argument("--maps-dir", type=str, required=True,
                        help="Directory containing feature map pkl files")
    parser.add_argument("--ae-weights-dir", type=str, required=True,
                        help="Directory with trained weights autoencoder")
    parser.add_argument("--ae-gradients-dir", type=str, required=True,
                        help="Directory with trained gradients autoencoder")
    parser.add_argument("--results-json", type=str, required=True,
                        help="Path to JSON file mapping arch IDs to final accuracies")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--split-json", type=str, required=True,
                        help="JSON with 'train' and 'val' keys listing architecture IDs")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Predictor hidden state size (default: 2048 for lstm, 128 for bigru)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sharpness-dir", type=str, default=None,
                        help="Directory with sharpness pkl files (optional, for F1-F4)")
    parser.add_argument("--sharpness-features", type=str, default=None,
                        help="Comma-separated sharpness features to use: F1,F2,F3,F4")
    parser.add_argument("--sharpness-only", action="store_true",
                        help="Use only sharpness features (no AE embeddings)")
    parser.add_argument("--aug", type=int, default=0,
                        help="Augmentation steps for anytime prediction (0=disabled, e.g. 10)")
    parser.add_argument("--predictor-type", type=str, default="lstm",
                        choices=["lstm", "bigru"],
                        help="Predictor architecture: lstm (original) or bigru (dual-path)")
    parser.add_argument("--dense-size", type=int, default=None,
                        help="Dense layer size for biGRU predictor (default: 128 for bigru, 64 for lstm)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for biGRU predictor (default: 0.1)")
    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["step", "onecycle"],
                        help="LR scheduler: step (WarmUp+StepLR) or onecycle (OneCycleLR)")
    parser.add_argument("--max-lr", type=float, default=3e-3,
                        help="Max LR for OneCycleLR scheduler (default: 3e-3)")
    args = parser.parse_args()

    # Resolve predictor-type-dependent defaults
    if args.hidden_size is None:
        args.hidden_size = 128 if args.predictor_type == "bigru" else 2048

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    maps_dir = Path(args.maps_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split — same train/val/test as AE
    with open(args.split_json) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val architectures")

    # Load sharpness data if provided
    sharpness_data = {}
    sharpness_feature_names = []
    if args.sharpness_dir:
        sharpness_feature_names = (
            args.sharpness_features.split(",") if args.sharpness_features
            else ["F1", "F2", "F3", "F4"]
        )
        sharpness_dir = Path(args.sharpness_dir)
        for sf_path in sharpness_dir.glob("*_sharpness.pkl"):
            aid = sf_path.stem.replace("_sharpness", "")
            with open(str(sf_path), "rb") as f:
                sharpness_data[aid] = pickle.load(f)
        print(f"Loaded sharpness data for {len(sharpness_data)} architectures")
        print(f"Using sharpness features: {sharpness_feature_names}")

    # Load autoencoders (skip if sharpness-only)
    ae_weights = None
    ae_gradients = None
    ae_normalize = "none"
    if not args.sharpness_only:
        ae_weights = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_weights_dir) / "best_ae_model.pt"),
            params_path=str(Path(args.ae_weights_dir) / "model_hyper_params.json"),
        )
        ae_gradients = FeatureMapAutoEncoder.load(
            model_path=str(Path(args.ae_gradients_dir) / "best_ae_model.pt"),
            params_path=str(Path(args.ae_gradients_dir) / "model_hyper_params.json"),
        )
        # Detect normalization mode from AE params
        ae_params_path = Path(args.ae_weights_dir) / "model_hyper_params.json"
        with open(ae_params_path) as f:
            ae_params = json.load(f)
        ae_normalize = ae_params.get("normalize", "none")
        if ae_normalize != "none":
            print(f"AE trained with normalize={ae_normalize}, applying to feature maps")

    # Load ground truth results
    with open(args.results_json) as f:
        results = json.load(f)

    # Encode feature maps
    print("Loading and encoding feature maps...")
    if args.sharpness_only:
        train_emb, train_tgt = _encode_sharpness_only(
            train_ids, results, sharpness_data, sharpness_feature_names)
        val_emb, val_tgt = _encode_sharpness_only(
            val_ids, results, sharpness_data, sharpness_feature_names)
    else:
        train_emb, train_tgt = _encode_architectures(
            train_ids, maps_dir, ae_weights, ae_gradients, results,
            sharpness_data, sharpness_feature_names, normalize=ae_normalize)
        val_emb, val_tgt = _encode_architectures(
            val_ids, maps_dir, ae_weights, ae_gradients, results,
            sharpness_data, sharpness_feature_names, normalize=ae_normalize)

    if not train_emb:
        print("No valid training data found. Exiting.")
        return

    has_val = len(val_emb) > 0
    print(f"Encoded {len(train_emb)} train, {len(val_emb)} val architecture sequences")

    # Z-score normalize sharpness features (computed on train, applied to val)
    n_sharpness = len(sharpness_feature_names)
    sharpness_stats = None
    if n_sharpness > 0:
        train_emb, sharpness_stats = _normalize_sharpness(train_emb, n_sharpness)
        if has_val:
            val_emb, _ = _normalize_sharpness(val_emb, n_sharpness, stats=sharpness_stats)
        print(f"Sharpness normalization (train stats, {n_sharpness} features):")
        for i, feat in enumerate(sharpness_feature_names):
            print(f"  {feat}: mean={sharpness_stats['mean'][i]:.4f}, "
                  f"std={sharpness_stats['std'][i]:.4f}")

    # Augment training sequences for anytime prediction (NOT val)
    if args.aug > 0:
        pre_aug = len(train_emb)
        train_emb, train_tgt = _augment_sequences(train_emb, train_tgt, args.aug)
        print(f"Augmentation (aug={args.aug}): {pre_aug} -> {len(train_emb)} "
              f"training sequences ({len(train_emb) // pre_aug}x)")

    # Create datasets
    print("Padding and stacking sequences...", flush=True)
    train_X, embedding_size = _pad_and_stack(train_emb)
    train_y = torch.tensor(train_tgt, dtype=torch.float64).unsqueeze(1)
    print(f"Train tensor: {train_X.shape} ({train_X.nbytes / 1e6:.1f} MB)", flush=True)
    train_loader = DataLoader(TensorDataset(train_X, train_y),
                              batch_size=args.batch_size, shuffle=True)

    if has_val:
        val_X, _ = _pad_and_stack(val_emb)
        val_y = torch.tensor(val_tgt, dtype=torch.float64).unsqueeze(1)
        print(f"Val tensor: {val_X.shape} ({val_X.nbytes / 1e6:.1f} MB)", flush=True)
        val_loader = DataLoader(TensorDataset(val_X, val_y),
                                batch_size=args.batch_size, shuffle=False)

    # Create predictor model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.predictor_type == "bigru":
        dense_size = args.dense_size if args.dense_size is not None else 128
        model = BiGRUDualPredictor(
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
            dense_size=dense_size,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LSTMPredictor(
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # L1Loss (MAE) matches original NAP2 repo (loss_name="mae")
    criterion = torch.nn.L1Loss()

    # LR schedule
    if args.scheduler == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=args.max_lr,
                               epochs=args.epochs,
                               steps_per_epoch=steps_per_epoch)
    else:
        # 5-epoch linear warmup (1e-6 -> 1e-3) then StepLR decay
        # Matches original NAP2 ModelTrainer protocol
        step_scheduler = StepLR(optimizer, step_size=20, gamma=0.15)
        scheduler = WarmUpLR(optimizer, step_scheduler,
                             warmup_epochs=5, initial_lr=1e-6, target_lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters, device={device}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"aug={args.aug}")
    print(f"Scheduler: WarmUpLR(5 epochs, 1e-6->1e-3) + "
          f"StepLR(step=20, gamma=0.15)")
    print(f"Train batches/epoch: {len(train_loader)}, "
          f"Val batches/epoch: {len(val_loader) if has_val else 0}")

    best_val_loss = float("inf")
    cp_dir = output_dir / "cp" / "model_state_cp"
    cp_dir.mkdir(parents=True, exist_ok=True)
    model_path = cp_dir / "model.pt"

    # Save params before training so they survive time-limit kills
    dense_size = args.dense_size if args.dense_size is not None else (128 if args.predictor_type == "bigru" else 64)
    params = {
        "predictor_type": args.predictor_type,
        "embedding_size": embedding_size,
        "hidden_size": args.hidden_size,
        "lstm_layers": 1 if args.predictor_type == "lstm" else None,
        "gru_layers": 2 if args.predictor_type == "bigru" else None,
        "inner_dense_layer_sizes": [64] if args.predictor_type == "lstm" else None,
        "dense_size": dense_size if args.predictor_type == "bigru" else None,
        "dropout": args.dropout if args.predictor_type == "bigru" else None,
        "last_layer": "sigmoid",
        "bi_directional": args.predictor_type == "bigru",
        "batch_first": True,
        "is_double": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "normalize": ae_normalize,
        "sharpness_features": sharpness_feature_names if sharpness_feature_names else None,
        "sharpness_only": args.sharpness_only,
        "sharpness_norm": {
            "mean": sharpness_stats["mean"].tolist(),
            "std": sharpness_stats["std"].tolist(),
        } if sharpness_stats else None,
        "aug": args.aug,
    }
    params_path = output_dir / "model_hyper_params.json"
    with open(str(params_path), "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params to {params_path}")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            if args.scheduler == "onecycle":
                scheduler.step()
        avg_train_loss = train_loss / train_batches

        # Validate
        if has_val:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches
        else:
            avg_val_loss = avg_train_loss

        # Best model checkpoint on val loss (overwrite)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(model_path))

        improved = " *" if avg_val_loss == best_val_loss else ""
        current_lr = optimizer.param_groups[0]['lr']
        if has_val:
            print(f"Epoch {epoch + 1}/{args.epochs}  "
                  f"train={avg_train_loss:.6f}  val={avg_val_loss:.6f}  "
                  f"best={best_val_loss:.6f}  lr={current_lr:.2e}{improved}")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}  "
                  f"loss={avg_train_loss:.6f}  best={best_val_loss:.6f}  "
                  f"lr={current_lr:.2e}{improved}")

        if args.scheduler != "onecycle":
            scheduler.step()

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
