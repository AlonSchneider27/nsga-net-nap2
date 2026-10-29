"""Spherical CIFAR-100 (NAS-Bench-360) loader for the search phase.

File: ``s2_cifar100.gz`` (gzip pickle) = {'train': {'images': uint8 [50000,3,60,60], 'labels': int64},
'test': {'images': uint8 [10000,3,60,60], 'labels': int64}}. Same NAS-style split as the CIFAR loaders:
``train=True`` -> the first 40k of the original train set, ``train=False`` -> the last 10k as validation.
The 10k test set is untouched (reserved for a ground-truth protocol).
"""
import gzip
import os
import pickle

import numpy as np
from PIL import Image
import torch.utils.data as data

FILENAME = 's2_cifar100.gz'
N_TRAIN = 40000


class SphericalCIFAR100(data.Dataset):
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root or 'data/spherical_cifar100'
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        path = os.path.join(self.root, FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} not found (NAS-Bench-360 s2_cifar100.gz; no auto-download)')
        with gzip.open(path, 'rb') as f:
            d = pickle.load(f)
        x, y = d['train']['images'], np.asarray(d['train']['labels'], dtype=np.int64)
        sl = slice(0, N_TRAIN) if train else slice(N_TRAIN, None)
        # keep HWC uint8 so PIL-based torchvision transforms apply unchanged
        self._data = np.ascontiguousarray(np.transpose(x[sl], (0, 2, 3, 1)))
        self._labels = y[sl]

    def __getitem__(self, index):
        img = Image.fromarray(self._data[index])
        target = int(self._labels[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self._labels)
