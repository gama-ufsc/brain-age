import h5py

from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pickle
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

        return self._get_item(dataset, index_)

    def _get_item(self, dataset, index_):
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
        get_age=False,
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

        self.get_age = get_age

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

    def fix_label(self, label):
        try:
            labels_i_orders = np.stack([self._labels_i_order,] * label.shape[0]).T
            relative_label = np.apply_along_axis(np.where, 0, labels_i_orders == label)
        except IndexError:
            relative_label = np.where(self._labels_i_order == label)

        return relative_label[0][0]

    def __getitem__(self, index: int):
        if self.get_age:
            img, age, label = super().__getitem__(self._idx[index])

            return img, age, self.fix_label(label)
        else:
            img, label = super().__getitem__(self._idx[index])

            return img, self.fix_label(label)

    def _get_item(self, dataset, index_):
        with h5py.File(self._fpath, 'r') as h:
            img = h[dataset]['X'][index_]
            if self.get_age:
                age = h[dataset]['a'][index_]
            label = h[dataset]['y'][index_]

        # transform
        if self.transform is not None:
            img = self.transform(img)

        if self.get_age:
            return img, age, label
        else:
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

class BraTSDataset(Dataset):
    def __init__(
        self,
        imgs_dir='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetData_plans_v2.1_2D_stage0/',
        meta_fpath='/home/jupyter/gama/nnUNet/data/raw/survival_info.csv',
        plans_fpath='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_FabiansResUNet_v2.1_plans_2D.pkl',
        transform=torch.Tensor,
    ) -> None:
        super().__init__()

        self.imgs_dir = Path(imgs_dir)

        assert self.imgs_dir.exists()

        self.meta = pd.read_csv(meta_fpath).set_index('Brats20ID')
        with open(plans_fpath, 'rb') as f:
            plans = pickle.load(f)
        self.patch_size = np.array([192, 192])

        self.transform = transform

    def __len__(self):
        return self.meta.shape[0] * 80

    def __getitem__(self, index: int):
        lb = 35
        ub = 115
        
        i = index // 80
        j = index % 80

        img_id = self.meta.iloc[i].name
        label = float(self.meta.iloc[i]['Age'])

        img_meta_fpath = self.imgs_dir/f"{img_id}.pkl"
        with open(img_meta_fpath, 'rb') as f:
            img_meta = pickle.load(f)
        crop_offset = img_meta['crop_bbox'][0][0]

        full_img_fpath = self.imgs_dir/f"{img_id}.npz"
        img = np.load(full_img_fpath)['data'][0, 35 - crop_offset + j]

        # transform
        if self.transform is not None:
            img = self.transform(img)

        # pad
        pad = self.patch_size - img.shape
        pad_left = (pad / 2).astype(int)
        pad_right = (0.5 + pad / 2).astype(int)

        f_pad = np.empty((pad_left.size + pad_right.size,), dtype=pad_left.dtype)
        f_pad[0::2] = pad_left
        f_pad[1::2] = pad_right

        img = torch.nn.functional.pad(img, f_pad.tolist()[::-1]).unsqueeze(0)[:,:192,:192]

        return img, label
