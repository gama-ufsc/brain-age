# Brain Age

## Preprocessing

The src directory contains two files for preprocessing:

1. `preprocess`: performs registration using the ANTs toolkit.
2. `make_dataset`: performs skull stripping using the HD-BET toolkit and converts images to the .hdf5 format.

## Trainer
The `src/trainer` directory contains two files for training:

1. `trainer_v1:` for training with images having only 1 channel, and only for the U-Net architecture used in this project.
2. `trainer_v2:` for training with images having 3 channels, used for pre-trained architectures in ImageNet.


The `experiments.py` files contain the training processes used, along with the model configurations.
There is also an `experiments_tuning_lr.py` file that contains information on hyperparameter tuning,
and an `experiments-final.py` file that contains the model re-trained on the split (80% - 20%).