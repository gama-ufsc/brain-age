import h5py

from pathlib import Path

import numpy as np
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

class ADNIDatasetClassification(ADNIDataset):
    def __init__(
        self,
        hdf5_fpath,
        transform=torch.Tensor,
        dataset='train',
        labels=['CN','AD'],
    ) -> None:
        super().__init__(
            hdf5_fpath=hdf5_fpath,
            transform=transform,
            dataset=dataset,
        )

        self._labels_order = np.array(['CN', 'EMCI', 'LMCI', 'MCI', 'AD', 'SMC'])
        self.labels = labels
        self._labels_i_order = np.where(np.isin(self._labels_order, self.labels))[0]

        self._update_idx()

    def _update_idx(self):
        with h5py.File(self._fpath, 'r') as h:
            # get true length
            if self.dataset == 'train+val':
                l = h['train']['y'].shape[0]
                l += h['val']['y'].shape[0]
            else:
                l = h[self.dataset]['y'].shape[0]

            y_is_labels = np.full((l,), False)
            for label in self.labels:
                label_i = np.where(self._labels_order == label)[0]

                if self.dataset == 'train+val':
                    y_is_label_train = h['train']['y'][:] == label_i
                    y_is_label_val = h['val']['y'][:] == label_i
                    y_is_label = np.concatenate([y_is_label_train, y_is_label_val])
                else:
                    y_is_label = h[self.dataset]['y'][:] == label_i

                y_is_labels |= y_is_label

        self._idx = np.where(y_is_labels)[0]

    def __len__(self):
        # Uncomment the line below if the dataset changes along the usage
#         self._update_idx()

        return len(self._idx)

    def __getitem__(self, index: int):
        img, label = super().__getitem__(self._idx[index])

        return img, np.where(self._labels_i_order == label)[0]

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
