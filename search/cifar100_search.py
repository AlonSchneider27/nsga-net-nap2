"""CIFAR-100 loader for the NAS search phase.

Mirrors search/cifar10_search.py but for CIFAR-100. Splits the 50k training
set deterministically into 40k (train) + 10k (valid). The original 10k test
set is intentionally unused at search time, matching the CIFAR-10 protocol.
"""

from __future__ import print_function

import errno
import hashlib
import os
import os.path
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    return md5o.hexdigest() == md5


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except Exception:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


class CIFAR100(data.Dataset):
    """CIFAR-100 with the NAS-style 40k/10k train/valid split.

    Args:
        root: directory containing (or that will receive) ``cifar-100-python/``.
        train: True for the 40k training shard; False for the 10k validation
            shard carved out of the original train set.
        transform/target_transform: callables applied to image / label.
        download: download the tarball if missing.
    """

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', '8a2e4ce0d5e60add47e7b9e9b7d4ddc3'],
    ]
    # number of items kept as the search-time "train" shard; the remainder
    # of the original 50k train set is used as the held-out validation shard.
    _train_split = 40000

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.'
                               ' You can use download=True to download it')

        # Load the single 50k train pickle and slice deterministically.
        f = self.train_list[0][0]
        path = os.path.join(self.root, self.base_folder, f)
        with open(path, 'rb') as fo:
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
        # CIFAR-100's pickle has 'fine_labels' (100-class) and 'coarse_labels' (20-class).
        labels = entry.get('fine_labels', entry.get('labels'))
        all_data = entry['data']  # shape (50000, 3072) uint8

        if self.train:
            sl = slice(0, self._train_split)
        else:
            sl = slice(self._train_split, None)

        data_arr = all_data[sl]
        n = data_arr.shape[0]
        data_arr = data_arr.reshape((n, 3, 32, 32))
        data_arr = data_arr.transpose((0, 2, 3, 1))  # to HWC
        labels_arr = list(labels[sl] if isinstance(labels, np.ndarray) else labels[sl.start:sl.stop])

        if self.train:
            self.train_data = data_arr
            self.train_labels = labels_arr
        else:
            self.test_data = data_arr
            self.test_labels = labels_arr

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), 'r:gz')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'valid'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


if __name__ == '__main__':
    ds = CIFAR100(root='data', train=True, download=True)
    print(ds)
