from subprocess import CalledProcessError
import h5py

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.preprocessing.preprocessing import PreprocessorFor2D
from tqdm import tqdm

from brats.preprocessing.hdbet_wrapper import hd_bet

import nibabel as nib
from nibabel.processing import conform


PREP_DATA_DIR = Path('/home/bruno-pacheco/brain-age/data/raw/ADNI_prep')
DATASET_FPATH = Path('/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix_2mm.hdf5')

SPLIT_CSV_FPATH = Path('/home/bruno-pacheco/brain-age/notebooks/dataframe2D-ADNI.csv')

DOWNSIZE = True

tmpdir = Path('.tmpdir')

def load_preprocess_for_nnunet(
        img_fpath,
        tmpdir='.tmpdir',
        patch_size=[192, 160],
        input_shape_must_be_divisible_by=[32, 32],
        normalization_schemes=OrderedDict([(0, 'nonCT')]),
        use_mask_for_norm=OrderedDict([(0, True)]),
        transpose_forward=[0, 1, 2],
        intensity_properties=None,
        target_spacing=[1., 1., 1.],
    ):
    brain_img_fpath, _ = hd_bet(img_fpath, tmpdir, mode='fast')

    prep = PreprocessorFor2D(normalization_schemes, use_mask_for_norm,
                                transpose_forward, intensity_properties)
    brain, _, prop = prep.preprocess_test_case([str(brain_img_fpath),], target_spacing)

    # get only slices with meaningful brain info
    crop_lb = prop['crop_bbox'][0][0]
    lb = int(35 / target_spacing[0]) - crop_lb
    ub = int(115 / target_spacing[0]) - crop_lb
    brain = brain[:,lb:ub]

    padded_brain, _ = pad_nd_image(
        brain,
        new_shape=patch_size,
        mode='constant',
        kwargs={'constant_values': 0},
        return_slicer=True,
        shape_must_be_divisible_by=input_shape_must_be_divisible_by,
    )

    return padded_brain

if __name__ == '__main__':
    assert PREP_DATA_DIR.exists(), f"`{PREP_DATA_DIR}` doesn't exist"
    assert DATASET_FPATH.parent.exists(), f"`{DATASET_FPATH.parent}` doesn't exist"

    tmpdir.mkdir(exist_ok=True)

    assert SPLIT_CSV_FPATH.exists(), f"`{SPLIT_CSV_FPATH}` doesn't exist"

    target_shape = (80, 192, 160)

    if DOWNSIZE:
        target_shape = (40, 96, 96)

    # split data
    df = pd.read_csv(SPLIT_CSV_FPATH)
    train_sids = df[df['split'] == 'train']['patient'].unique()

    train_fpaths_ = [list(PREP_DATA_DIR.glob(f"{sid[5:]}*.nii")) for sid in train_sids]
    train_fpaths = list()
    for fs in train_fpaths_:
        train_fpaths += fs
    train_fpaths = sorted(train_fpaths)

    test_fpaths = list(set(PREP_DATA_DIR.glob('*.nii')).difference(set(train_fpaths)))
    test_fpaths = sorted(test_fpaths)

    i_train = 0
    i_test = 0

    # create dataset
    if DATASET_FPATH.exists():
        # check if there's any progress already
        with h5py.File(DATASET_FPATH, 'r') as h:
            # overwrite the last image just to be sure
            n_train = (h['train']['y'].shape[0] // target_shape[0]) - 1
            n_test = (h['test']['y'].shape[0] // target_shape[0]) -1 

        train_fpaths = train_fpaths[n_train-1:]
        test_fpaths = test_fpaths[n_test-1:]

        i_train = n_train * target_shape[0]
        i_test = n_test * target_shape[0]
    else:
        with h5py.File(DATASET_FPATH, 'w') as h:
            train = h.create_group('train')
            X_train = train.create_dataset(
                'X',
                (0,target_shape[1],target_shape[2]),
                maxshape=(None,target_shape[1],target_shape[2]),
                dtype='float32',
                chunks=(1,target_shape[1],target_shape[2]),
                compression='gzip',
            )
            y_train = train.create_dataset(
                'y',
                (0,),
                maxshape=(None,),
                dtype='uint8',
            )

            test = h.create_group('test')
            X_test = test.create_dataset(
                'X',
                (0,target_shape[1],target_shape[2]),
                maxshape=(None,target_shape[1],target_shape[2]),
                dtype='float32',
                chunks=(1,target_shape[1],target_shape[2]),
                compression='gzip',
            )
            y_test = test.create_dataset(
                'y',
                (0,),
                maxshape=(None,),
                dtype='uint8',
            )

    def update_dataset(imgs_fpaths, i, ds_name):
        for img_fpath in tqdm(imgs_fpaths):
            age = int(img_fpath.name.split('.')[0].split('__')[-1])

            img = nib.load(img_fpath)
            dsz_img = conform(img, out_shape=tuple(np.array(img.shape) // 2), voxel_size=(2.,2.,2.))
            img_fpath = tmpdir/img_fpath.name
            nib.save(dsz_img, str(img_fpath))

            try:
                brain = load_preprocess_for_nnunet(img_fpath, patch_size=[96,80], target_spacing=[2., 2., 2.], tmpdir=str(tmpdir.resolve()))
            except CalledProcessError:
                print("WARNING!")
                # img_fpath.unlink()
                continue

            if brain[0].shape == target_shape:
                with h5py.File(DATASET_FPATH, 'r+') as h:
                    X = h[ds_name]['X']
                    y = h[ds_name]['y']

                    X.resize(X.shape[0] + target_shape[0], axis=0)
                    y.resize(y.shape[0] + target_shape[0], axis=0)

                    X[i:i+target_shape[0]] = brain[0]
                    y[i:i+target_shape[0]] = age
                i += target_shape[0]
            else:
                img_fpath.unlink()

    print('Working on test images')
    update_dataset(test_fpaths, i_test, 'test')

    print('Working on train images')
    update_dataset(train_fpaths, i_train, 'train')
