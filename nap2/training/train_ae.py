"""Train the feature map autoencoder.

Loads feature map pickle files, trains a FeatureMapAutoEncoder with early
stopping on validation loss, and saves the best model checkpoint.

Requires --split-json with 'train' and 'val' keys listing architecture IDs.
This ensures AE trains only on designated architectures with no data leakage.
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

from nap2.autoencoder import DEFAULT_LAYER_SHAPES, FeatureMapAutoEncoder
from nap2.feature_maps import log_normalize
from nap2.training.warmup import WarmUpLR


def _load_maps_to_tensor(maps_dir: Path, data_type: str,
                         arch_ids: list[str],
                         normalize: str = "none") -> torch.Tensor | None:
    """Load feature maps directly into a pre-allocated tensor.

    Returns a float32 tensor of shape [N, 12, 65, 100] where N = num_archs * num_snapshots.
    Pre-allocates to avoid intermediate copies that cause OOM.
    """
    # First pass: count valid architectures and snapshots
    sorted_ids = sorted(arch_ids)
    valid_ids = []
    for arch_id in sorted_ids:
        if (maps_dir / f"{arch_id}_{data_type}_maps.pkl").exists():
            valid_ids.append(arch_id)
    if not valid_ids:
        return None

    # Peek at first file to get snapshot count and shape
    with open(str(maps_dir / f"{valid_ids[0]}_{data_type}_maps.pkl"), "rb") as f:
        sample = pickle.load(f)
    n_snapshots = sample.shape[0]  # typically 31
    total_maps = len(valid_ids) * n_snapshots

    # Pre-allocate single tensor (no intermediate copies)
    print(f"  Pre-allocating tensor: [{total_maps}, 12, 65, 100] float32 "
          f"({total_maps * 12 * 65 * 100 * 4 / 1e9:.1f} GB)", flush=True)
    tensor = torch.zeros(total_maps, 12, 65, 100, dtype=torch.float32)

    # Second pass: load directly into tensor
    write_idx = 0
    for i, arch_id in enumerate(valid_ids):
        with open(str(maps_dir / f"{arch_id}_{data_type}_maps.pkl"), "rb") as f:
            maps = pickle.load(f)
        # maps: [n_snapshots, 65, 100, 12] -> reorder to [n_snapshots, 12, 65, 100]
        reordered = np.moveaxis(maps, -1, 1).astype(np.float32)
        np.nan_to_num(reordered, nan=0.0, posinf=1e7, neginf=-1e7, copy=False)
        if normalize == "log":
            reordered = log_normalize(reordered)
        tensor[write_idx:write_idx + n_snapshots] = torch.from_numpy(reordered)
        write_idx += n_snapshots
        if (i + 1) % 100 == 0 or i + 1 == len(valid_ids):
            print(f"  Loaded {i + 1}/{len(valid_ids)} architectures", flush=True)

    return tensor


def main():
    parser = argparse.ArgumentParser(description="Train feature map autoencoder")
    parser.add_argument("--maps-dir", type=str, required=True,
                        help="Directory containing feature map pkl files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trained model")
    parser.add_argument("--data-type", type=str, required=True,
                        choices=["weights", "gradients"],
                        help="Type of feature maps to train on")
    parser.add_argument("--split-json", type=str, required=True,
                        help="JSON with 'train' and 'val' keys listing architecture IDs")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--input-channels", type=int, default=12,
                        help="Number of input channels (stats). Default 12.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--normalize", type=str, default="none",
                        choices=["none", "log"],
                        help="Feature map normalization before AE training. "
                             "'log' applies log(1+|x|)*sign(x) for cross-dataset transfer.")
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    maps_dir = Path(args.maps_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split — architecture IDs for train and val
    with open(args.split_json) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"]
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val architectures")

    # Load feature maps directly into pre-allocated tensors (no intermediate copies)
    if args.normalize != "none":
        print(f"Feature map normalization: {args.normalize}")
    print(f"Loading {args.data_type} maps for train set...", flush=True)
    train_tensor = _load_maps_to_tensor(maps_dir, args.data_type, train_ids,
                                        normalize=args.normalize)
    print(f"Loading {args.data_type} maps for val set...", flush=True)
    val_tensor = _load_maps_to_tensor(maps_dir, args.data_type, val_ids,
                                      normalize=args.normalize)

    if train_tensor is None or val_tensor is None:
        print("Insufficient data. Exiting.")
        return

    print(f"Train: {train_tensor.shape}, Val: {val_tensor.shape}", flush=True)

    train_loader = DataLoader(TensorDataset(train_tensor),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor),
                            batch_size=args.batch_size, shuffle=False)

    # Create and train model (float32 for GPU efficiency; old repo used float64 on CPU
    # but float64 is ~32x slower on consumer GPUs — precision difference is negligible for KT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureMapAutoEncoder(input_channels=args.input_channels).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # reduction='sum' matches original NAP2 repo
    criterion = torch.nn.MSELoss(reduction='sum')

    # LR schedule: 5-epoch linear warmup (1e-6 -> lr) then StepLR decay
    # Matches original NAP2 ModelTrainer protocol
    step_scheduler = StepLR(optimizer, step_size=20, gamma=0.15)
    scheduler = WarmUpLR(optimizer, step_scheduler,
                         warmup_epochs=5, initial_lr=1e-6, target_lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters, device={device}, dtype=float32")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience}")
    print(f"Scheduler: WarmUpLR(5 epochs, 1e-6->{args.lr}) + "
          f"StepLR(step=20, gamma=0.15)")
    print(f"Train batches/epoch: {len(train_loader)}, "
          f"Val batches/epoch: {len(val_loader)}")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / train_batches

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches

        # Best model checkpoint on val loss (overwrite)
        improved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(output_dir / "best_ae_model.pt"))
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs}  "
              f"train={avg_train_loss:.6f}  val={avg_val_loss:.6f}  "
              f"best={best_val_loss:.6f}  patience={patience_counter}/{args.patience}  "
              f"lr={current_lr:.2e}{improved}")

        scheduler.step()

        # Early stopping on val loss
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    # Save hyperparameters
    params_path = output_dir / "model_hyper_params.json"
    params = {
        "layers_shapes": DEFAULT_LAYER_SHAPES,
        "input_channels": args.input_channels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_type": args.data_type,
        "seed": args.seed,
        "normalize": args.normalize,
    }
    with open(str(params_path), "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved best model to {output_dir / 'best_ae_model.pt'}")
    print(f"Saved params to {params_path}")


if __name__ == "__main__":
    main()
