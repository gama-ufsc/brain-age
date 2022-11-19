from subprocess import CalledProcessError
import h5py

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.preprocessing.preprocessing import PreprocessorFor2D
from tqdm import tqdm
from sklearn.model_selection import KFold

from brats.preprocessing.hdbet_wrapper import hd_bet

import nibabel as nib
from nibabel.processing import conform


PREP_DATA_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI1_prep')
DATASET_FPATH = Path('/home/jupyter/gama/bruno/data/interim/ADNI1_slices_fix_2mm_5fold.hdf5')

# SPLIT_CSV_FPATH = Path('/home/bruno-pacheco/brain-age/notebooks/dataframe3D.csv')

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

#     assert SPLIT_CSV_FPATH.exists(), f"`{SPLIT_CSV_FPATH}` doesn't exist"

    target_shape = (80, 192, 160)

    with open(PREP_DATA_DIR/'ages.pkl', 'rb') as f:
        ages = pickle.load(f)

    if DOWNSIZE:
        target_shape = (40, 96, 96)

#     # split data
#     df = pd.read_csv(SPLIT_CSV_FPATH)
#     train_sids = df[df['split'] == 'train']['patient'].unique()
#     val_sids = df[df['split'] == 'val']['patient'].unique()
#     test_sids = df[df['split'] == 'test']['patient'].unique()

#     train_fpaths_ = [list(PREP_DATA_DIR.glob(f"{sid[5:]}*.nii")) for sid in train_sids]
#     train_fpaths = list()
#     for fs in train_fpaths_:
#         train_fpaths += fs  # add all images from the subjects
#     train_fpaths = sorted(train_fpaths)

#     val_fpaths_ = [list(PREP_DATA_DIR.glob(f"{sid[5:]}*.nii")) for sid in val_sids]
#     val_fpaths = list()
#     for fs in val_fpaths_:
#         val_fpaths += fs  # add all images from the subjects
#     val_fpaths = sorted(val_fpaths)

#     test_fpaths_ = [list(PREP_DATA_DIR.glob(f"{sid[5:]}*.nii")) for sid in test_sids]
#     test_fpaths = list()
#     for fs in test_fpaths_:
#         test_fpaths += fs  # add all images from the subjects
#     test_fpaths = sorted(test_fpaths)

#     i_train = 0
#     i_test = 0
#     i_val = 0

    all_fpaths = list(PREP_DATA_DIR.glob(f"*.nii"))

    assert len(ages.keys()) == len(all_fpaths)

    # make splits
    splitter = KFold(5, shuffle=True, random_state=42)

    sids = np.array(list({path.name.split('__')[0] for path in all_fpaths}))

    folds_fpaths = list()
    for _, fold_ix in splitter.split(sids):
        fold_sids = sids[fold_ix]

        fold_fpaths = np.array([path for path in all_fpaths if path.name.split('__')[0] in fold_sids])

        folds_fpaths.append(fold_fpaths)

    i_folds = [0,] * len(folds_fpaths)

    # create dataset
    if DATASET_FPATH.exists():
        # check if there's any progress already
        for i, fold_fpaths in enumerate(folds_fpaths):
            with h5py.File(DATASET_FPATH, 'r') as h:
                # overwrite the last image just to be sure
                n = max((h[str(i)]['y'].shape[0] // target_shape[0]) - 1,0)

            fold_fpaths = fold_fpaths[n-1:]

            folds_fpaths[i] = (train_fpaths, test_fpaths)
            i_folds[i] = n * target_shape[0]
    else:
        with h5py.File(DATASET_FPATH, 'w') as h:            
            for i, _ in enumerate(folds_fpaths):
                fold = h.create_group(str(i))

                X = fold.create_dataset(
                    'X',
                    (0,target_shape[1],target_shape[2]),
                    maxshape=(None,target_shape[1],target_shape[2]),
                    dtype='float32',
                    chunks=(1,target_shape[1],target_shape[2]),
                    compression='gzip',
                )
                y = fold.create_dataset(
                    'y',
                    (0,),
                    maxshape=(None,),
                    dtype='float32',
                )

    def update_dataset(imgs_fpaths, i, ds_name):
        for img_fpath in tqdm(imgs_fpaths):
            age = ages[img_fpath.name.split('.')[0].split('__')[-1]]

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

                    X.resize(i + target_shape[0], axis=0)
                    y.resize(i + target_shape[0], axis=0)

                    X[i:i+target_shape[0]] = brain[0]
                    y[i:i+target_shape[0]] = age
                i += target_shape[0]
            else:
                img_fpath.unlink()

    for i, fold_fpaths in enumerate(folds_fpaths):
        print(f"Working on fold {i}")
        update_dataset(fold_fpaths, i_folds[i], str(i))
