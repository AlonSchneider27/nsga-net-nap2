"""Train NAS-Bench-101 models and collect weight/gradient snapshots.

Training protocol based on NAS-Bench-101 paper (Ying et al., 2019):
- RMSprop(lr=0.1, momentum=0.9, eps=1.0, weight_decay=1e-4)
- Cosine annealing LR per step: lr(t) = 0.5 * base_lr * (1 + cos(pi * t / T))
- CIFAR-10 with pad-4 → random crop 32×32, random horizontal flip, normalize
- Batch size 128 (NAP2 convention; NB-101 paper uses 256)
- Model: stem(3→128) → 3 stacks × 3 modules → GAP → Dense(num_classes)

Known deviations from exact NB-101 protocol:
- PyTorch RMSprop differs from TensorFlow RMSprop (nasbench_pytorch warns about this)
- Uses full 50K CIFAR-10 train set (NB-101 paper uses 40K train + 10K val split)
- Batch size 128 vs paper's 256

Uses nasbench_pytorch for model building from (adjacency_matrix, ops) specs.
Labels (108-epoch accuracy) come from the NB-101 benchmark, not from this training.

Usage:
    python -m nap2.training.train_snapshots_nb101 \\
        --search-space architectures_nb101.json \\
        --dataset-dir /path/to/cifar10 \\
        --output-dir ./snapshots_nb101 \\
        --index 1 --workers 10
"""
from __future__ import annotations

import argparse
import bz2
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nap2.snapshot_collector import SnapshotCollector


# ============================================================================
# NAS-Bench-101 model building (via nasbench_pytorch)
# ============================================================================

def build_nb101_model(matrix: list, ops: list, num_classes: int = 10,
                      stem_out_channels: int = 128,
                      num_stacks: int = 3,
                      num_modules_per_stack: int = 3) -> nn.Module:
    """Build a NAS-Bench-101 PyTorch model from adjacency matrix and ops.

    Args:
        matrix: 7×7 upper-triangular adjacency matrix (list of lists or np.ndarray)
        ops: list of 7 operation strings, e.g.
             ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3",
              "conv3x3-bn-relu", "conv1x1-bn-relu", "output"]
        num_classes: number of output classes (10 for CIFAR-10)
        stem_out_channels: initial channel count (128 per NB-101 paper)
        num_stacks: number of stacked modules (3 per NB-101 paper)
        num_modules_per_stack: modules per stack (3 per NB-101 paper)

    Returns:
        nn.Module that takes (B, 3, 32, 32) → (B, num_classes)
    """
    from nasbench_pytorch.model import Network as NB101Network
    from nasbench_pytorch.model import ModelSpec

    matrix_np = np.array(matrix, dtype=int)
    spec = ModelSpec(matrix=matrix_np, ops=ops)

    model = NB101Network(
        spec,
        num_labels=num_classes,
        in_channels=3,
        stem_out_channels=stem_out_channels,
        num_stacks=num_stacks,
        num_modules_per_stack=num_modules_per_stack,
    )

    # Disable in-place ReLU — required for gradient capture
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model


# ============================================================================
# NAS-Bench-101 training protocol
# ============================================================================

class CosineDecayPerStep:
    """Per-step cosine LR decay matching NAS-Bench-101 protocol.

    lr(t) = 0.5 * base_lr * (1 + cos(pi * t / T))

    This is applied per mini-batch step, not per epoch.
    """

    def __init__(self, optimizer, base_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps

    def step(self, current_step: int):
        progress = current_step / self.total_steps
        new_lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def load_cifar10_nb101(dataset_dir: str, batch_size: int = 128) -> DataLoader:
    """Load CIFAR-10 with NAS-Bench-101 augmentation protocol.

    Protocol from NB-101 paper (Ying et al. 2019):
    - Random horizontal flip
    - Pad 4 pixels, random crop to 32×32
    - Normalize per-channel: mean=[125.31, 122.95, 113.87]/255,
                              std=[62.99, 62.09, 66.70]/255
    """
    import torchvision
    import torchvision.transforms as T

    # NB-101 uses specific CIFAR-10 normalization (from the paper)
    mean = [125.31 / 255.0, 122.95 / 255.0, 113.87 / 255.0]
    std = [62.99 / 255.0, 62.09 / 255.0, 66.70 / 255.0]

    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, download=False, transform=transform,
    )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )


def load_fake_data(batch_size: int = 128) -> DataLoader:
    """Create fake CIFAR-10-like data for testing."""
    from torch.utils.data import TensorDataset
    inputs = torch.randn(batch_size * 20, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size * 20,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ============================================================================
# Snapshot collection
# ============================================================================

def collect_single_architecture(
    arch_id: str,
    arch_spec: dict,
    dataset_dir: str | None,
    output_dir: str,
    snapshot_interval: int = 100,
    max_snapshots: int = 31,
    num_classes: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.1,
    training_epochs: int = 36,
    use_fake_data: bool = False,
    device: str = "cpu",
) -> None:
    """Collect snapshots for a single NB-101 architecture.

    Training protocol matches NAS-Bench-101 paper:
    - RMSprop(lr=0.1, momentum=0.9, epsilon=1.0, weight_decay=1e-4)
    - Cosine LR decay per step
    - 36 epochs, batch_size=128
    - CIFAR-10 with NB-101 augmentation

    Saves:
        {output_dir}/{arch_id}_snapshots.bz2  -- weight/gradient snapshots
    """
    output_path = Path(output_dir)
    snapshot_file = output_path / f"{arch_id}_snapshots.bz2"

    # Build model
    matrix = arch_spec["matrix"]
    ops = arch_spec["ops"]
    model = build_nb101_model(matrix, ops, num_classes=num_classes)
    model = model.to(device)

    # Count parameters and layers for logging
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    # Load data
    if use_fake_data:
        dataloader = load_fake_data(batch_size)
    else:
        dataloader = load_cifar10_nb101(dataset_dir, batch_size)

    # Training setup — NB-101 protocol
    # eps=1.0 matches NB-101 paper (unusually large; PyTorch default is 1e-8)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        eps=1.0,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(dataloader)
    total_steps = training_epochs * steps_per_epoch
    lr_scheduler = CosineDecayPerStep(optimizer, learning_rate, total_steps)

    # Snapshot collector
    collector = SnapshotCollector(
        model, interval=snapshot_interval, max_snapshots=max_snapshots,
    )

    # Training loop
    model.train()
    global_step = 0
    max_steps = max_snapshots * snapshot_interval  # Stop after enough snapshots

    print(f"    params={n_params:,}, conv={n_conv}, linear={n_linear}, "
          f"steps_per_epoch={steps_per_epoch}, "
          f"total_steps={total_steps}, collecting {max_snapshots} snapshots "
          f"(every {snapshot_interval} steps, need {max_steps} steps)")

    for epoch in range(training_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # LR update (per step, before optimizer step — matches NB-101)
            lr_scheduler.step(global_step)

            # Forward + backward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Snapshot
            collector.step()
            global_step += 1

            # Early stop once we have enough snapshots
            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

        # Log epoch progress
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{training_epochs}, step {global_step}, "
                  f"lr={current_lr:.6f}, loss={loss.item():.4f}")

    # Save snapshots
    snapshots = collector.get_snapshots()
    n_collected = len(snapshots)
    print(f"    Collected {n_collected} snapshots, saving to {snapshot_file}")

    with bz2.open(str(snapshot_file), "wb") as f:
        pickle.dump(snapshots, f)


# ============================================================================
# Architecture loading
# ============================================================================

def load_nb101_architectures(search_space_file: str) -> list:
    """Load NB-101 architecture specs from JSON.

    Expected format:
    {
        "nb101_00000": {"matrix": [[0,1,1,...], ...], "ops": ["input", "conv3x3-bn-relu", ...]},
        "nb101_00001": {...},
        ...
    }

    Returns: list of (arch_id, arch_spec) pairs
    """
    with open(search_space_file) as f:
        data = json.load(f)
    return list(data.items())


def partition_files(files, index, total):
    """Partition file list for SLURM workers. index is 1-based."""
    chunk_size = len(files) // total
    start = (index - 1) * chunk_size
    end = start + chunk_size if index < total else len(files)
    return files[start:end]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NAS-Bench-101 models and collect snapshots"
    )
    parser.add_argument("--search-space", type=str, required=True,
                        help="JSON file with NB-101 architecture specs")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to CIFAR-10 dataset directory (required unless --use-fake-data)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save snapshot files")
    parser.add_argument("--index", type=int, default=1,
                        help="Worker index (1-based)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total number of workers")
    parser.add_argument("--snapshot-interval", type=int, default=100,
                        help="Mini-batches between snapshots")
    parser.add_argument("--max-snapshots", type=int, default=31,
                        help="Maximum snapshots to collect per architecture")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Base learning rate for RMSprop (NB-101 default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size (NB-101 default: 128)")
    parser.add_argument("--training-epochs", type=int, default=108,
                        help="Training epochs for cosine LR schedule horizon (108 = NB-101 labels budget)")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--use-fake-data", action="store_true",
                        help="Use fake data for testing (no CIFAR-10 needed)")
    parser.add_argument("--device", type=str, default="cuda"
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.use_fake_data and args.dataset_dir is None:
        parser.error("--dataset-dir is required unless --use-fake-data is set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architectures = load_nb101_architectures(args.search_space)
    architectures = partition_files(architectures, args.index, args.workers)

    print(f"Worker {args.index}/{args.workers}: processing {len(architectures)} architectures")
    print(f"Protocol: RMSprop lr={args.learning_rate}, batch={args.batch_size}, "
          f"epochs={args.training_epochs}, snapshots={args.max_snapshots}×{args.snapshot_interval}")

    for arch_id, arch_spec in architectures:
        snapshot_file = output_dir / f"{arch_id}_snapshots.bz2"

        if snapshot_file.exists():
            print(f"  Skipping {arch_id} (already exists)")
            continue

        print(f"  Training {arch_id}...")
        try:
            collect_single_architecture(
                arch_id=arch_id,
                arch_spec=arch_spec,
                dataset_dir=args.dataset_dir,
                output_dir=str(output_dir),
                snapshot_interval=args.snapshot_interval,
                max_snapshots=args.max_snapshots,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                training_epochs=args.training_epochs,
                use_fake_data=args.use_fake_data,
                device=args.device,
            )
            print(f"  Done with {arch_id}")
        except Exception as e:
            print(f"  ERROR on {arch_id}: {e}")
            continue

    print("All architectures complete.")


if __name__ == "__main__":
    main()
