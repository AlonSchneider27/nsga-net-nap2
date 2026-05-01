"""ImageNet16-120 loader for the NAS search phase.

NB201's downsampled ImageNet variant: 16x16 RGB images, 120 classes, ~151k
train + ~6k val. Distributed by the NB201 paper authors as a tarball
containing four .npy files: ``x_train.npy``, ``y_train.npy``, ``x_val.npy``,
``y_val.npy``.

Auto-download flow mirrors search/cifar10_search.py:
    - If the .npy files already exist under ``root``, load them.
    - Otherwise, fetch a tarball from ``IMAGENET16_URL`` (constant below;
      override via the ``IMAGENET16_URL`` env var) and extract.
    - On total failure, raise FileNotFoundError naming both the expected
      paths and the URL that was attempted, with manual-download instructions.

Default mirror is left blank because ImageNet16-120 has no canonical public
URL — pick a working mirror (HuggingFace dataset, GitHub release, etc.) and
either set the ``IMAGENET16_URL`` env var or paste it into ``DEFAULT_URL``
below.
"""

from __future__ import print_function

import errno
import hashlib
import os
import os.path
import sys
import tarfile

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


# Override at runtime via the IMAGENET16_URL env var, or edit this default.
DEFAULT_URL: str = ''

# MD5 of the tarball expected at DEFAULT_URL. Empty string disables
# integrity check (download still verified by file existence only).
DEFAULT_MD5: str = ''

EXPECTED_FILES = ('x_train.npy', 'y_train.npy', 'x_val.npy', 'y_val.npy')


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
    expected = '\n  '.join(os.path.join(root, n) for n in EXPECTED_FILES)
    msg = (
        'ImageNet16-120 data files not found. Expected:\n  '
        f'{expected}\n'
        f"\nDownload was attempted from: {url or '(no URL configured)'}\n"
        '\nTo fix:\n'
        '  1. Set the IMAGENET16_URL env var to a working mirror, or paste\n'
        '     a URL into DEFAULT_URL in search/imagenet16_search.py.\n'
        '  2. Or download the tarball manually and extract the four .npy\n'
        f'     files into {root}/.\n'
        '\nThe NB201 ImageNet16-120 dataset is distributed via the original\n'
        'NB201 paper authors (https://github.com/D-X-Y/AutoDL-Projects).\n'
    )
    return FileNotFoundError(msg)


class ImageNet16(data.Dataset):
    """NB201 ImageNet16-120: 16x16 RGB, 120 classes.

    Args:
        root: directory containing (or that will receive) the four .npy files.
        train: True -> x_train.npy/y_train.npy; False -> x_val.npy/y_val.npy.
        transform/target_transform: callables applied to image / label.
        download: if True and files are missing, fetch a tarball from
            ``IMAGENET16_URL`` env var or ``DEFAULT_URL``.
        url: optional explicit download URL (overrides env var and default).
        md5: optional MD5 to verify the tarball.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 url: str = '', md5: str = ''):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

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

        suffix = 'train' if self.train else 'val'
        x = np.load(os.path.join(self.root, f'x_{suffix}.npy'))
        y = np.load(os.path.join(self.root, f'y_{suffix}.npy'))

        # Some distributions store x as float32 in [0,1] CHW; some store uint8
        # HWC. Coerce to uint8 HWC so downstream PIL conversion works.
        if x.dtype != np.uint8:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max <= 1.0 + 1e-6:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
            # CHW -> HWC
            x = np.transpose(x, (0, 2, 3, 1))

        self._data = x
        self._labels = np.asarray(y).astype(np.int64).tolist()

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
    ds = ImageNet16(root='data/ImageNet16-120', train=True, download=True)
    print(ds)
