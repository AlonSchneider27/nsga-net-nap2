"""Central dataset registry shared by search and validation phases.

A single source of truth for class count, normalization stats, image size,
and which loader module/class to instantiate. New datasets get added here
plus a matching loader module under ``search/``.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

import torchvision.transforms as transforms


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'cifar10': {
        'num_classes': 10,
        'image_size': (32, 32),
        'mean': [0.49139968, 0.48215827, 0.44653124],
        'std':  [0.24703233, 0.24348505, 0.26158768],
        'data_dir': 'data',
        'loader_module': 'search.cifar10_search',
        'loader_class': 'CIFAR10',
    },
    'cifar100': {
        'num_classes': 100,
        'image_size': (32, 32),
        'mean': [0.5071, 0.4867, 0.4408],
        'std':  [0.2675, 0.2565, 0.2761],
        'data_dir': 'data',
        'loader_module': 'search.cifar100_search',
        'loader_class': 'CIFAR100',
    },
    'ImageNet16-120': {
        'num_classes': 120,
        'image_size': (16, 16),
        'mean': None,   # NB201 protocol: no normalization
        'std':  None,
        # Default root; override per-run with --data (the loader
        # auto-detects the .npy layout vs the NB201 pickle batches).
        'data_dir': 'data/ImageNet16',
        'loader_module': 'search.imagenet16_search',
        'loader_class': 'ImageNet16',
    },
}


def get_config(dataset: str) -> Dict[str, Any]:
    """Return the config dict for a registered dataset name."""
    if dataset not in DATASET_CONFIGS:
        valid = sorted(DATASET_CONFIGS.keys())
        raise KeyError(f"Unknown dataset {dataset!r}. Valid choices: {valid}")
    return DATASET_CONFIGS[dataset]


def get_loader_class(dataset: str):
    """Import and return the Dataset class registered for ``dataset``."""
    cfg = get_config(dataset)
    mod = importlib.import_module(cfg['loader_module'])
    return getattr(mod, cfg['loader_class'])


def build_search_transforms(cfg: Dict[str, Any], train: bool):
    """Compose torchvision transforms for the search-phase dataloaders.

    Mirrors what ``train_search.py`` did inline for CIFAR-10: random crop +
    horizontal flip + ToTensor + (optional) Normalize. ``Normalize`` is
    skipped when ``cfg['mean']`` is None (NB201 protocol for ImageNet16-120).
    """
    h, w = cfg['image_size']
    # CIFAR uses padding=4 around 32x32; for 16x16 NB201 uses padding=2.
    padding = 4 if h >= 32 else 2

    pieces = []
    if train:
        pieces.append(transforms.RandomCrop(h, padding=padding))
        pieces.append(transforms.RandomHorizontalFlip())
    pieces.append(transforms.ToTensor())
    if cfg['mean'] is not None and cfg['std'] is not None:
        pieces.append(transforms.Normalize(cfg['mean'], cfg['std']))
    return transforms.Compose(pieces)
