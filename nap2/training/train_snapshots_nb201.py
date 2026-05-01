"""Train NAS-Bench-201 models and collect snapshots + sharpness features.

This is the NB201-specific version of train_snapshots.py that:
1. Builds models using the standalone NB201 builder (no xautodl needed)
2. Loads CIFAR-10 with NB201-default transforms
3. Collects weight/gradient snapshots via SnapshotCollector
4. Computes F1-F4 sharpness features at each snapshot
"""
from __future__ import annotations

import argparse
import bz2
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nap2.search_spaces.nb201_ops import build_nb201_model
from nap2.sharpness import compute_sharpness_features
from nap2.snapshot_collector import SnapshotCollector


def partition_files(files, index, total):
    """Partition file list for SLURM workers. index is 1-based."""
    chunk_size = len(files) // total
    start = (index - 1) * chunk_size
    end = start + chunk_size if index < total else len(files)
    return files[start:end]


def load_dataset(dataset_dir: str, dataset_name: str = "cifar10",
                  batch_size: int = 256) -> DataLoader:
    """Load dataset with NAS-Bench-201 default transforms.

    Matches protocol from nasbench201_model_builder_script.py exactly:
    - CIFAR-10: Normalize(0.4914,0.4822,0.4465), (0.247,0.243,0.261)
    - CIFAR-100: Normalize(0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
    - ImageNet-16-120: NO normalization, loads from .npy files, 16x16 images

    Supports: cifar10, cifar100, ImageNet16-120.
    """
    import torchvision
    import torchvision.transforms as T

    if dataset_name == "cifar10":
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir, train=True, download=True, transform=transform,
        )

    elif dataset_name == "cifar100":
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        dataset = torchvision.datasets.CIFAR100(
            root=dataset_dir, train=True, download=True, transform=transform,
        )

    elif dataset_name.lower().startswith("imagenet16"):
        # ImageNet-16-120: loads from .npy files, NO normalization
        # Matches nasbench201_model_builder_script.py protocol
        from torch.utils.data import Dataset
        x = np.load(str(Path(dataset_dir) / "x_train.npy"))
        y = np.load(str(Path(dataset_dir) / "y_train.npy"))
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(16, padding=2),
            T.ToTensor(),
        ])

        class ImageNet16Dataset(Dataset):
            def __init__(self, x, y, transform):
                self.x, self.y, self.transform = x, y, transform
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                img = T.ToPILImage()(self.x[idx])
                return self.transform(img), self.y[idx]

        dataset = ImageNet16Dataset(x, y, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def load_fake_data(batch_size: int = 256) -> DataLoader:
    """Create fake CIFAR-10-like data for testing."""
    from torch.utils.data import TensorDataset

    inputs = torch.randn(batch_size * 20, 3, 32, 32)  # Enough for ~20 batches
    targets = torch.randint(0, 10, (batch_size * 20,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def collect_single_architecture(
    arch_id: str,
    arch_str: str,
    dataset_dir: str | None,
    output_dir: str,
    snapshot_interval: int = 100,
    max_snapshots: int = 31,
    dataset_name: str = "cifar10",
    learning_rate: float = 0.1,
    num_classes: int = 10,
    batch_size: int = 256,
    use_fake_data: bool = False,
    device: str = "cpu",
) -> None:
    """Collect snapshots and sharpness features for a single architecture.

    Saves two files:
        {output_dir}/{arch_id}_snapshots.bz2  -- weight/gradient snapshots
        {output_dir}/{arch_id}_sharpness.pkl   -- F1-F4 per snapshot
    """
    output_path = Path(output_dir)
    snapshot_file = output_path / f"{arch_id}_snapshots.bz2"
    sharpness_file = output_path / f"{arch_id}_sharpness.pkl"

    # Build model
    model = build_nb201_model(arch_str, num_classes=num_classes)
    model = model.to(device)

    # Load data
    if use_fake_data:
        dataloader = load_fake_data(batch_size)
    else:
        dataloader = load_dataset(dataset_dir, dataset_name=dataset_name, batch_size=batch_size)

    # Setup training
    collector = SnapshotCollector(
        model, interval=snapshot_interval, max_snapshots=max_snapshots,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9,
        nesterov=True, weight_decay=0.0005,
    )
    criterion = nn.CrossEntropyLoss()

    # CosineAnnealingLR — matches NB-201 training protocol (T_max=200)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Sharpness tracking
    sharpness_data = {"F1": [], "F2": [], "F3": [], "F4": [],
                      "F2_converged": [], "snapshots": []}
    prev_lambda_h = None
    batch_count = 0
    batches_per_epoch = len(dataloader)

    model.train()
    data_iter = iter(dataloader)
    total_batches = max_snapshots * snapshot_interval

    while batch_count < total_batches:
        # Get next batch (cycle if exhausted = new epoch, step scheduler)
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            scheduler.step()
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[-1]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Snapshot collection
        collector.step()
        batch_count += 1

        # Compute sharpness at snapshot points
        if batch_count % snapshot_interval == 0 and len(sharpness_data["F1"]) < max_snapshots:
            features = compute_sharpness_features(
                model, criterion, inputs, targets,
                learning_rate=learning_rate,
                prev_lambda_h=prev_lambda_h,
                snapshot_interval=snapshot_interval,
            )

            sharpness_data["F1"].append(features["F1"])
            sharpness_data["F2"].append(features["F2"])
            sharpness_data["F3"].append(features["F3"])
            sharpness_data["F4"].append(features["F4"])
            sharpness_data["F2_converged"].append(features["F2_converged"])
            sharpness_data["snapshots"].append(batch_count)

            prev_lambda_h = features["F2"]

            print(f"    Snapshot {len(sharpness_data['F1'])}/{max_snapshots}: "
                  f"F1={features['F1']:.4f}, F2={features['F2']:.4f}, "
                  f"F3={features['F3']:.4f}, converged={features['F2_converged']}")

    # Save snapshots
    snapshots = collector.get_snapshots()
    with bz2.open(str(snapshot_file), "wb") as f:
        pickle.dump(snapshots, f)

    # Save sharpness
    with open(str(sharpness_file), "wb") as f:
        pickle.dump(sharpness_data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Train NB201 models and collect snapshots + sharpness features"
    )
    parser.add_argument("--search-space", type=str, required=True,
                        help="JSON file: {arch_id: arch_string}")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to CIFAR-10 dataset directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--index", type=int, default=1,
                        help="Worker index (1-based)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total workers")
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--max-snapshots", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--dataset-name", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "ImageNet16-120"],
                        help="Dataset to train on (affects data loader and transforms)")
    parser.add_argument("--device", type=str, default="cuda"
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.search_space) as f:
        search_space = json.load(f)

    architectures = list(search_space.items())
    architectures = partition_files(architectures, args.index, args.workers)

    print(f"Worker {args.index}/{args.workers}: processing {len(architectures)} architectures")

    for arch_id, arch_str in architectures:
        snapshot_file = output_dir / f"{arch_id}_snapshots.bz2"
        sharpness_file = output_dir / f"{arch_id}_sharpness.pkl"

        if snapshot_file.exists() and sharpness_file.exists():
            print(f"  Skipping {arch_id} (already exists)")
            continue

        print(f"  Training {arch_id}...")
        collect_single_architecture(
            arch_id=arch_id,
            arch_str=arch_str,
            dataset_dir=args.dataset_dir,
            output_dir=str(output_dir),
            snapshot_interval=args.snapshot_interval,
            max_snapshots=args.max_snapshots,
            dataset_name=args.dataset_name,
            learning_rate=args.learning_rate,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"  Done with {arch_id}")

    print("All architectures complete.")


if __name__ == "__main__":
    main()
