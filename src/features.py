from pathlib import Path

import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import ADNIDatasetForBraTSModel
from net import BraTSnnUNet


SLICES_DATASET_FPATH = Path('/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5')
FEATURES_DATASET_FPATH = Path('/home/bruno-pacheco/brain-age/data/interim/ADNI_features_brats_fix.hdf5')

if __name__ == '__main__':
    # load slices
    train_data = ADNIDatasetForBraTSModel(SLICES_DATASET_FPATH, dataset='train', transform=transforms.ToTensor())
    test_data = ADNIDatasetForBraTSModel(SLICES_DATASET_FPATH, dataset='test', transform=transforms.ToTensor())

    n_train = len(train_data)
    n_test = len(test_data)

    train_dataloader = DataLoader(train_data, batch_size=2**8, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=2**8, shuffle=False)

    # create features hdf5
    with h5py.File(FEATURES_DATASET_FPATH, 'w') as h:
        train = h.create_group('train')
        X_train = train.create_dataset(
            'X',
            (n_train,480,6,5),
            dtype='float32',
            chunks=(1,480,6,5),
            # compression='gzip',
        )
        y_train = train.create_dataset(
            'y',
            (n_train,),
            dtype='uint8',
        )

        test = h.create_group('test')
        X_test = test.create_dataset(
            'X',
            (n_test,480,6,5),
            dtype='float32',
            chunks=(1,480,6,5),
            # compression='gzip',
        )
        y_test = test.create_dataset(
            'y',
            (n_test,),
            dtype='uint8',
        )

    # load features encoder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        *torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt').brats_encoder
    ).to(device)

    net.eval()
    with torch.no_grad():
        i = 0
        for X, y in tqdm(train_dataloader):
            n_slices = X.shape[0]
            X, y = X.to(device), y.to(device)

            features = net(X)

            with h5py.File(FEATURES_DATASET_FPATH, 'r+') as h:
                X_features = h['train']['X']
                y_features = h['train']['y']

                X_features[i:i + n_slices] = features.cpu().detach().numpy()
                y_features[i:i + n_slices] = y.cpu().detach().numpy()

            i += n_slices

        i = 0
        for X, y in tqdm(test_dataloader):
            n_slices = X.shape[0]
            X, y = X.to(device), y.to(device)

            features = net(X)

            with h5py.File(FEATURES_DATASET_FPATH, 'r+') as h:
                X_features = h['test']['X']
                y_features = h['test']['y']

                X_features[i:i + n_slices] = features.cpu().detach().numpy()
                y_features[i:i + n_slices] = y.cpu().detach().numpy()

            i += n_slices
