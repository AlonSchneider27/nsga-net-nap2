"""Extract statistical features from weight/gradient snapshots.

Loads bz2-compressed snapshot files, computes statistics via extract_all_stats,
and saves results as bz2-compressed pickle files.
"""
from __future__ import annotations

import argparse
import bz2
import pickle
from pathlib import Path

from nap2.stats import extract_all_stats


def partition_files(files, index, total):
    """Partition file list for SLURM workers. index is 1-based."""
    chunk_size = len(files) // total
    start = (index - 1) * chunk_size
    end = start + chunk_size if index < total else len(files)
    return files[start:end]


def main():
    parser = argparse.ArgumentParser(description="Extract statistics from snapshots")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing snapshot bz2 files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for stats files")
    parser.add_argument("--index", type=int, default=1,
                        help="Worker index (1-based) for SLURM partitioning")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total number of SLURM workers")
    parser.add_argument("--data-type", type=str, required=True,
                        choices=["weights", "gradients"],
                        help="Type of data to extract stats from")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all snapshot files
    all_files = sorted(input_dir.glob("*_snapshots.bz2"))
    if not all_files:
        print(f"No snapshot files found in {input_dir}")
        return

    # Partition for this worker
    files = partition_files(all_files, args.index, args.workers)
    print(f"Worker {args.index}/{args.workers}: processing {len(files)} files")

    for snapshot_path in files:
        arch_id = snapshot_path.stem.replace("_snapshots", "")
        output_path = output_dir / f"{arch_id}_{args.data_type}_stats.bz2"

        if output_path.exists():
            print(f"  Skipping {arch_id} (already exists)")
            continue

        print(f"  Extracting {args.data_type} stats for {arch_id}...")

        with bz2.open(str(snapshot_path), "rb") as f:
            snapshots = pickle.load(f)

        # Extract only the requested data type
        if args.data_type == "weights":
            typed_snapshots = {
                step: {name: data["weights"] for name, data in layers.items()}
                for step, layers in snapshots.items()
            }
        else:
            typed_snapshots = {
                step: {name: data["gradients"] for name, data in layers.items()}
                for step, layers in snapshots.items()
            }

        stats = extract_all_stats(typed_snapshots)

        with bz2.open(str(output_path), "wb") as f:
            pickle.dump(stats, f)

        print(f"  Saved stats for {arch_id}")

    print("Done.")


if __name__ == "__main__":
    main()
