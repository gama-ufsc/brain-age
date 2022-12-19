import h5py

from pathlib import Path

import torch

from torch.utils.data import Dataset
from torchvision import transforms


class ADNIDataset(Dataset):
    def __init__(
        self,
        hdf5_fpath,
        transform=torch.Tensor,
        dataset='train',
    ) -> None:
        super().__init__()

        hdf5_fpath = Path(hdf5_fpath)
        assert hdf5_fpath.exists()

        self._fpath = hdf5_fpath
        self.dataset = dataset

        self.transform = transform

    def __len__(self):
        with h5py.File(self._fpath, 'r') as h:
            if self.dataset == 'train+val':
                l = h['train']['y'].shape[0]
                l += h['val']['y'].shape[0]
            else:
                l = h[self.dataset]['y'].shape[0]

        return l

    def __getitem__(self, index: int):
        if self.dataset == 'train+val':
            with h5py.File(self._fpath, 'r') as h:
                train_len = h['train']['y'].shape[0]
            index_ = index - train_len

            if index_ < 0:
                dataset = 'train'
            else:
                dataset = 'val'
        else:
            index_ = index
            dataset = self.dataset

        with h5py.File(self._fpath, 'r') as h:
            img = h[dataset]['X'][index_]
            label = h[dataset]['y'][index_]

        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class ADNIDatasetKFold(Dataset):
    def __init__(
        self,
        hdf5_fpath,
        transform=torch.Tensor,
        dataset='train',
        split=0,
    ) -> None:
        super().__init__()

        hdf5_fpath = Path(hdf5_fpath)
        assert hdf5_fpath.exists()

        self._fpath = hdf5_fpath
        
        assert dataset.lower() in ['train', 'test'], '`dataset` must be either `train` or `test`.'
        self.dataset = dataset.lower()
        self.split = split

        self.transform = transform

    def __len__(self):
        with h5py.File(self._fpath, 'r') as h:
            if self.dataset == 'test':
                l = h[str(self.split)]['y'].shape[0]
            else:
                l = 0
                for i in range(5):
                    if i != self.split:
                        l += h[str(i)]['y'].shape[0]

        return l

    def __getitem__(self, index: int):
        if self.dataset == 'train':
            splits = [i for i in range(5) if i != self.split]
            sizes = list()
            for i in splits:
                with h5py.File(self._fpath, 'r') as h:
                    sizes.append(h[str(i)]['y'].shape[0])

            full_len = self.__len__()
            assert sum(sizes) == full_len
            if index >= full_len:
                raise IndexError

            for i, size in enumerate(sizes):
                index = index - size
                split = splits[i]
                if index < 0:
                    break
        else:
            split = self.split

        with h5py.File(self._fpath, 'r') as h:
            img = h[str(split)]['X'][index]
            label = h[str(split)]['y'][index]

        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
class ADNISemiSupervisedDataset(ADNIDataset):
    def __getitem__(self, index: int):
        with h5py.File(self._fpath, 'r') as h:
            img = h[str(self.split)]['X'][index]
            label = h[str(self.split)]['y'][index]

        i_slice = index % 40

        # transform
        if self.transform is not None:
            img = self.transform(img)

        return img, label, i_slice
