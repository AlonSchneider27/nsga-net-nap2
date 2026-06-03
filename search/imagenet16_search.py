"""ImageNet16-120 loader for the NAS search phase.

NB201's downsampled ImageNet variant: 16x16 RGB images, 120 classes, ~151k
train + ~6k val. Distributed by the NB201 paper authors as a tarball
containing eleven Python pickle files: ``train_data_batch_1`` through
``train_data_batch_10`` (sharded train set) plus ``val_data`` (the full
validation set). Each pickle is a dict like::

    {'data': numpy.ndarray of shape (N, 3*16*16) uint8,
     'labels': list[int]}                # 1-indexed in the canonical dist

We concatenate the train shards and remap labels to 0..119 to match
PyTorch CrossEntropy's expectations (and what xautodl's reference loader
does).

Alternative ``.npy`` layout: if ``x_train.npy`` / ``y_train.npy`` (and
``x_val.npy`` / ``y_val.npy``) are present under ``root`` they take
precedence over the pickle batches and are loaded directly — this is the
format nap2 reads (see nap2/training/train_snapshots_nb201.py). No download
is attempted in that case.

Auto-download flow mirrors search/cifar10_search.py:
    - If the eleven batch files already exist under ``root``, load them.
    - Otherwise, fetch a tarball from ``IMAGENET16_URL`` (constant below;
      override via the ``IMAGENET16_URL`` env var) and extract.
    - On total failure, raise FileNotFoundError naming both the expected
      paths and the URL that was attempted, with manual-download instructions.

------------------------------------------------------------------
HARDCODE YOUR DATASET PATH HERE if you don't want to set --data:
edit DEFAULT_ROOT below or the matching DATASET_CONFIGS entry in
misc/dataset_configs.py.
------------------------------------------------------------------
"""

from __future__ import print_function

import errno
import hashlib
import os
import os.path
import pickle
import sys
import tarfile

import numpy as np
import torch.utils.data as data
from PIL import Image


# Paste your absolute path here if it's not the default. The config
# registry in misc/dataset_configs.py reads this constant so the change
# propagates to evolution_search.py / validation/{train,test}.py.
DEFAULT_ROOT: str = 'data/ImageNet16'

# Override at runtime via the IMAGENET16_URL env var, or edit this default.
DEFAULT_URL: str = ''

# MD5 of the tarball expected at DEFAULT_URL. Empty string disables
# integrity check (download still verified by file existence only).
DEFAULT_MD5: str = ''

TRAIN_BATCHES = tuple(f'train_data_batch_{i}' for i in range(1, 11))
VAL_FILE = 'val_data'
EXPECTED_FILES = TRAIN_BATCHES + (VAL_FILE,)

# Alternative .npy layout (x_train/y_train/x_val/y_val), as written by the
# nap2 data pipeline. Loaded directly when present; takes precedence over the
# NB201 pickle batches.
NPY_TRAIN_FILES = ('x_train.npy', 'y_train.npy')
NPY_VAL_FILES = ('x_val.npy', 'y_val.npy')


def _check_md5(fpath: str, expected_md5: str) -> bool:
    if not expected_md5:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    return md5o.hexdigest() == expected_md5


def _download_and_extract(url: str, root: str, md5: str) -> None:
    """Download a tarball into ``root`` and extract it in place."""
    from six.moves import urllib

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    filename = os.path.basename(url) or 'imagenet16.tar.gz'
    fpath = os.path.join(root, filename)

    if os.path.isfile(fpath) and _check_md5(fpath, md5):
        print(f'Using downloaded file: {fpath}')
    else:
        print(f'Downloading {url} to {fpath}')
        urllib.request.urlretrieve(url, fpath)
        if md5 and not _check_md5(fpath, md5):
            raise RuntimeError(f'MD5 mismatch for {fpath}; expected {md5}')

    cwd = os.getcwd()
    try:
        os.chdir(root)
        with tarfile.open(fpath, 'r:*') as tar:
            tar.extractall()
    finally:
        os.chdir(cwd)


def _all_files_present(root: str) -> bool:
    return all(os.path.isfile(os.path.join(root, name)) for name in EXPECTED_FILES)


def _missing_files_error(root: str, url: str) -> FileNotFoundError:
    expected_lines = [os.path.join(root, n) for n in EXPECTED_FILES]
    expected = '\n  '.join(expected_lines)
    msg = (
        'ImageNet16-120 data files not found. Expected the eleven NB201\n'
        'batch files:\n  '
        f'{expected}\n'
        f"\nDownload was attempted from: {url or '(no URL configured)'}\n"
        '\nTo fix:\n'
        '  1. Set the IMAGENET16_URL env var to a working mirror, or paste\n'
        '     a URL into DEFAULT_URL in search/imagenet16_search.py.\n'
        '  2. Or download the tarball manually and extract the eleven batch\n'
        f'     files into {root}/.\n'
        '  3. Or hardcode your absolute path by editing DEFAULT_ROOT in\n'
        '     search/imagenet16_search.py (and the matching data_dir in\n'
        '     misc/dataset_configs.py).\n'
        '\nThe NB201 ImageNet16-120 dataset is distributed via the original\n'
        'NB201 paper authors (https://github.com/D-X-Y/AutoDL-Projects).\n'
    )
    return FileNotFoundError(msg)


def _load_nb201_pickle(path: str):
    """Load one NB201 batch pickle.

    Returns:
        data: ndarray (N, 3*16*16) uint8
        labels: list[int]; remapped from the canonical 1..120 to 0..119
    """
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    raw = entry['data']                            # (N, 3*16*16) uint8
    labels = [int(y) - 1 for y in entry['labels']]
    return raw, labels


def _npy_files_present(root: str, train: bool) -> bool:
    names = NPY_TRAIN_FILES if train else NPY_VAL_FILES
    return all(os.path.isfile(os.path.join(root, n)) for n in names)


def _load_npy_split(root: str, train: bool):
    """Load ImageNet16-120 from the .npy layout.

    Mirrors the format nap2 reads (nap2/training/train_snapshots_nb201.py:
    load_dataset). Robust to the image array being stored flat ((N, 768)),
    channel-first ((N, 3, 16, 16)), or channel-last ((N, 16, 16, 3)), and to
    float ([0, 1] or [0, 255]) vs uint8 pixels.

    Returns:
        data:   ndarray (N, 16, 16, 3) uint8 (HWC, for PIL.Image.fromarray)
        labels: list[int], 0-indexed
    """
    suffix = 'train' if train else 'val'
    x = np.load(os.path.join(root, f'x_{suffix}.npy'))
    y = np.load(os.path.join(root, f'y_{suffix}.npy'))

    x = np.asarray(x)
    if x.ndim == 2:                         # flat (N, 3*16*16) -> (N, 3, 16, 16)
        x = x.reshape(-1, 3, 16, 16)
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        x = x.transpose(0, 2, 3, 1)         # CHW -> HWC for PIL
    if np.issubdtype(x.dtype, np.floating):
        if float(x.max()) <= 1.0 + 1e-6:    # floats in [0, 1] -> [0, 255]
            x = x * 255.0
        x = np.clip(x, 0, 255).round().astype(np.uint8)
    else:
        x = x.astype(np.uint8)

    labels = [int(v) for v in np.asarray(y).ravel().tolist()]
    if labels and min(labels) == 1:         # defensive: 1..120 -> 0..119
        labels = [v - 1 for v in labels]
    return x, labels


class ImageNet16(data.Dataset):
    """NB201 ImageNet16-120: 16x16 RGB, 120 classes.

    Args:
        root: directory containing (or that will receive) the eleven
            NB201 batch files. Defaults to ``DEFAULT_ROOT``.
        train: True -> ``train_data_batch_1..10``;
            False -> ``val_data``.
        transform/target_transform: callables applied to image / label.
        download: if True and files are missing, fetch a tarball from
            ``IMAGENET16_URL`` env var or ``DEFAULT_URL``.
        url: optional explicit download URL (overrides env var and default).
        md5: optional MD5 to verify the tarball.
    """

    def __init__(self, root=None, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 url: str = '', md5: str = ''):
        if root is None:
            root = DEFAULT_ROOT
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # Prefer the .npy layout (x_train/y_train/x_val/y_val) when present;
        # it's the format nap2 writes/reads. No download in this case.
        if _npy_files_present(self.root, self.train):
            self._data, self._labels = _load_npy_split(self.root, self.train)
            return

        resolved_url = url or os.environ.get('IMAGENET16_URL', '') or DEFAULT_URL
        resolved_md5 = md5 or DEFAULT_MD5

        if download and not _all_files_present(self.root):
            if not resolved_url:
                raise _missing_files_error(self.root, resolved_url)
            try:
                _download_and_extract(resolved_url, self.root, resolved_md5)
            except Exception as e:
                # Surface the underlying network/extraction error context but
                # raise the higher-level FileNotFoundError with full guidance.
                print(f'ImageNet16-120 download failed: {e}')

        if not _all_files_present(self.root):
            raise _missing_files_error(self.root, resolved_url)

        # ---- load the NB201 pickle batches -----------------------------
        if self.train:
            chunks = []
            labels = []
            for name in TRAIN_BATCHES:
                d, y = _load_nb201_pickle(os.path.join(self.root, name))
                chunks.append(d)
                labels.extend(y)
            x = np.concatenate(chunks, axis=0)
        else:
            x, labels = _load_nb201_pickle(os.path.join(self.root, VAL_FILE))

        # NB201 ships uint8 (N, 768). Reshape to (N, 3, 16, 16) and
        # transpose to HWC for PIL.Image.fromarray.
        n = x.shape[0]
        x = x.reshape(n, 3, 16, 16).transpose(0, 2, 3, 1).astype(np.uint8)

        self._data = x
        self._labels = labels

    def __getitem__(self, index):
        img, target = self._data[index], self._labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self._labels)

    # Aliases so dispatching code can use the same attribute names as the
    # CIFAR loaders (where train_data/test_data are public).
    @property
    def train_data(self):
        return self._data if self.train else None

    @property
    def test_data(self):
        return self._data if not self.train else None

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


if __name__ == '__main__':
    ds = ImageNet16(root=DEFAULT_ROOT, train=True, download=True)
    print(ds)
