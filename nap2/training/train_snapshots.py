"""Train models and collect weight/gradient snapshots.

Partitions architectures across SLURM workers, trains each model using
SnapshotCollector, and saves snapshots as bz2-compressed pickle files.
"""
from __future__ import annotations

import argparse
import bz2
import json
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from nap2.snapshot_collector import SnapshotCollector


def partition_files(files, index, total):
    """Partition file list for SLURM workers. index is 1-based."""
    chunk_size = len(files) // total
    start = (index - 1) * chunk_size
    end = start + chunk_size if index < total else len(files)
    return files[start:end]


def main():
    parser = argparse.ArgumentParser(description="Train models and collect snapshots")
    parser.add_argument("--search-space", type=str, required=True,
                        help="Path to search space JSON file with architecture definitions")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for snapshot files")
    parser.add_argument("--index", type=int, default=1,
                        help="Worker index (1-based) for SLURM partitioning")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total number of SLURM workers")
    parser.add_argument("--max-models", type=int, default=None,
                        help="Maximum number of models to process")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs per model")
    parser.add_argument("--snapshot-interval", type=int, default=100,
                        help="Mini-batches between snapshots")
    parser.add_argument("--max-snapshots", type=int, default=23,
                        help="Maximum number of snapshots per model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load search space
    with open(args.search_space) as f:
        search_space = json.load(f)

    architectures = list(search_space.items())
    if args.max_models is not None:
        architectures = architectures[:args.max_models]

    # Partition for this worker
    architectures = partition_files(architectures, args.index, args.workers)

    print(f"Worker {args.index}/{args.workers}: processing {len(architectures)} architectures")

    for arch_id, arch_config in architectures:
        output_path = output_dir / f"{arch_id}_snapshots.bz2"
        if output_path.exists():
            print(f"  Skipping {arch_id} (already exists)")
            continue

        print(f"  Training {arch_id}...")

        # Build model from config (search-space specific)
        # The search space JSON should contain model configs that can be instantiated
        model = _build_model(arch_config)
        dataloader = _load_dataset(args.dataset_dir)

        collector = SnapshotCollector(
            model,
            interval=args.snapshot_interval,
            max_snapshots=args.max_snapshots,
        )

        # Train
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9,
            nesterov=True, weight_decay=0.0005,
        )
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(args.epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                collector.step()

                if len(collector.get_snapshots()) >= args.max_snapshots:
                    break
            if len(collector.get_snapshots()) >= args.max_snapshots:
                break

        snapshots = collector.get_snapshots()
        with bz2.open(str(output_path), "wb") as f:
            pickle.dump(snapshots, f)

        print(f"  Saved {len(snapshots)} snapshots for {arch_id}")

    print("Done.")


def _build_model(arch_config):
    """Build a model from architecture config. Override for your search space."""
    raise NotImplementedError(
        "Implement _build_model for your specific search space. "
        "This function should return an nn.Module from the architecture config dict."
    )


def _load_dataset(dataset_dir):
    """Load dataset and return a DataLoader. Override for your dataset."""
    raise NotImplementedError(
        "Implement _load_dataset for your specific dataset. "
        "This function should return a torch DataLoader."
    )


if __name__ == "__main__":
    main()
