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


ADNI1_DATA_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI1_prep')
ADNI23_DATA_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI23_prep')
DATASET_FPATH = Path('/home/jupyter/gama/bruno/data/interim/ADNI123_slices_fix_2mm_split.hdf5')

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
    assert ADNI1_DATA_DIR.exists(), f"`{ADNI1_DATA_DIR}` doesn't exist"
    assert ADNI23_DATA_DIR.exists(), f"`{ADNI23_DATA_DIR}` doesn't exist"
    assert DATASET_FPATH.parent.exists(), f"`{DATASET_FPATH.parent}` doesn't exist"
    
    tmpdir.mkdir(exist_ok=True)

#     assert SPLIT_CSV_FPATH.exists(), f"`{SPLIT_CSV_FPATH}` doesn't exist"

    target_shape = (80, 192, 160)

    with open(ADNI1_DATA_DIR/'groups.pkl', 'rb') as f:
        adni1_groups = pickle.load(f)

    with open(ADNI1_DATA_DIR/'ages.pkl', 'rb') as f:
        adni1_ages = pickle.load(f)

    
    with open(ADNI23_DATA_DIR/'groups.pkl', 'rb') as f:
        adni23_groups = pickle.load(f)

    with open(ADNI23_DATA_DIR/'ages.pkl', 'rb') as f:
        adni23_ages = pickle.load(f)

    ages = dict()
    for k,v in adni1_ages.items():
        ages[k] = v
    for k,v in adni23_ages.items():
        ages[k] = v

    if DOWNSIZE:
        target_shape = (40, 96, 96)

    adni1_fpaths = list(ADNI1_DATA_DIR.glob(f"*.nii"))
    adni23_fpaths = list(ADNI23_DATA_DIR.glob(f"*.nii"))

    assert len(adni1_ages.keys()) == len(adni1_fpaths)
    assert len(adni23_ages.keys()) == len(adni23_fpaths)

    # pick only CN
    adni1_fpaths = [fp for fp in adni1_fpaths if adni1_groups[fp.name.split('__')[1].rstrip('.nii')] == 'CN']
    adni23_fpaths = [fp for fp in adni23_fpaths if adni23_groups[fp.name.split('__')[1].rstrip('.nii')] == 'CN']

    # split data
    df_cv = pd.read_csv('/home/jupyter/gama/bruno/data/external/CROSSVAL.csv')
    df_cv = df_cv.set_index('RID')

    train_fpaths = adni23_fpaths
    
    val_fpaths = [fp for fp in adni1_fpaths if df_cv.loc[int(fp.name.split('__')[0].split('_S_')[1])]['TRAINING'] == 1]
    
    test_fpaths = [fp for fp in adni1_fpaths if df_cv.loc[int(fp.name.split('__')[0].split('_S_')[1])]['TRAINING'] == 0]

    i_train = 0
    i_test = 0
    i_val = 0

#     # make splits
#     splitter = KFold(5, shuffle=True, random_state=42)

#     sids = np.array(list({path.name.split('__')[0] for path in all_fpaths}))

#     folds_fpaths = list()
#     for _, fold_ix in splitter.split(sids):
#         fold_sids = sids[fold_ix]

#         fold_fpaths = np.array([path for path in all_fpaths if path.name.split('__')[0] in fold_sids])

#         folds_fpaths.append(fold_fpaths)

#     i_folds = [0,] * len(folds_fpaths)

    # create dataset
    if DATASET_FPATH.exists():
        # check if there's any progress already
        with h5py.File(DATASET_FPATH, 'r') as h:
            # always overwrite the last image just to be sure
            n_train = max((h['train']['y'].shape[0] // target_shape[0]) - 1,0)
            n_val = max((h['val']['y'].shape[0] // target_shape[0]) - 1,0)
            n_test = max((h['test']['y'].shape[0] // target_shape[0]) - 1,0)

        train_fpaths = train_fpaths[n_train-1:]
        val_fpaths = val_fpaths[n_val-1:]
        test_fpaths = test_fpaths[n_test-1:]

        i_train = n_train * target_shape[0]
        i_val = n_val * target_shape[0]
        i_test = n_test * target_shape[0]
    else:
        with h5py.File(DATASET_FPATH, 'w') as h:
            train = h.create_group('train')
            X = train.create_dataset(
                'X',
                (0,target_shape[1],target_shape[2]),
                maxshape=(None,target_shape[1],target_shape[2]),
                dtype='float32',
                chunks=(1,target_shape[1],target_shape[2]),
                compression='gzip',
            )
            y = train.create_dataset(
                'y',
                (0,),
                maxshape=(None,),
                dtype='float32',
            )

            val = h.create_group('val')
            X = val.create_dataset(
                'X',
                (0,target_shape[1],target_shape[2]),
                maxshape=(None,target_shape[1],target_shape[2]),
                dtype='float32',
                chunks=(1,target_shape[1],target_shape[2]),
                compression='gzip',
            )
            y = val.create_dataset(
                'y',
                (0,),
                maxshape=(None,),
                dtype='float32',
            )

            test = h.create_group('test')
            X = test.create_dataset(
                'X',
                (0,target_shape[1],target_shape[2]),
                maxshape=(None,target_shape[1],target_shape[2]),
                dtype='float32',
                chunks=(1,target_shape[1],target_shape[2]),
                compression='gzip',
            )
            y = test.create_dataset(
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

    for ds in ['train', 'val', 'test']:
        print(f"Working on {ds} dataset")
        update_dataset(eval(f"{ds}_fpaths"), eval(f"i_{ds}"), ds)
