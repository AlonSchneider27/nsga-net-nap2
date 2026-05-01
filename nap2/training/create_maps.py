"""Create feature maps from statistical features.

Loads bz2-compressed stats files, creates feature map sequences via
create_feature_map_sequence, and saves results as pickle files.
"""
from __future__ import annotations

import argparse
import bz2
import pickle
from pathlib import Path

from nap2.feature_maps import create_feature_map_sequence


def partition_files(files, index, total):
    """Partition file list for SLURM workers. index is 1-based."""
    chunk_size = len(files) // total
    start = (index - 1) * chunk_size
    end = start + chunk_size if index < total else len(files)
    return files[start:end]


def main():
    parser = argparse.ArgumentParser(description="Create feature maps from stats")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing stats bz2 files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for feature map files")
    parser.add_argument("--index", type=int, default=1,
                        help="Worker index (1-based) for SLURM partitioning")
    parser.add_argument("--workers", type=int, default=1,
                        help="Total number of SLURM workers")
    parser.add_argument("--data-type", type=str, required=True,
                        choices=["weights", "gradients"],
                        help="Type of data to create feature maps for")
    parser.add_argument("--stat-names", type=str, default=None,
                        help="Comma-separated stat names to include (default: all 12)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all stats files for the requested data type
    all_files = sorted(input_dir.glob(f"*_{args.data_type}_stats.bz2"))
    if not all_files:
        print(f"No {args.data_type} stats files found in {input_dir}")
        return

    # Partition for this worker
    files = partition_files(all_files, args.index, args.workers)
    print(f"Worker {args.index}/{args.workers}: processing {len(files)} files")

    for stats_path in files:
        arch_id = stats_path.stem.replace(f"_{args.data_type}_stats", "")
        output_path = output_dir / f"{arch_id}_{args.data_type}_maps.pkl"

        if output_path.exists():
            print(f"  Skipping {arch_id} (already exists)")
            continue

        print(f"  Creating {args.data_type} feature maps for {arch_id}...")

        with bz2.open(str(stats_path), "rb") as f:
            stats = pickle.load(f)

        stat_names = args.stat_names.split(",") if args.stat_names else None
        feature_maps = create_feature_map_sequence(stats, stat_names=stat_names)

        with open(str(output_path), "wb") as f:
            pickle.dump(feature_maps, f)

        print(f"  Saved feature maps for {arch_id} (shape: {feature_maps.shape})")

    print("Done.")


if __name__ == "__main__":
    main()
