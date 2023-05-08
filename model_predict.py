import sys
import pickle
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

from tqdm import tqdm

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision import transforms, models
from torch.utils.data import DataLoader
from nnunet.training.model_restore import restore_model
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import nnUNetTrainerV2_ResNetUNet

# local imports
from src.data import ADNIDatasetClassification
from src.net import BraTSnnUNet, load_from_wandb

PROJ_ROOT = Path('/home/jupyter/gama/bruno')
DATASET_FPATH = PROJ_ROOT/'data/interim/ADNI123_slices_fix_2mm_split_class.hdf5'


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PARSE ARGS
    args = sys.argv    
    model, run_id, split = args[-3:]

    model = model.lower()
    split = split.lower()

    assert model in ['unet', 'resnet']
    assert split in ['val', 'test']

    # LOAD DATA
    group_labels = ['CN', 'MCI', 'AD']
    dataset = ADNIDatasetClassification(
        DATASET_FPATH,
        get_age=True,
        dataset=split,
        labels=group_labels,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),

    )
    dataloader = DataLoader(dataset, batch_size=40, shuffle=False)

    h = lambda x: x*25+75

    # LOAD MODEL
    if split == 'val':
        wandb_model_file = 'model_best'
    elif split == 'test':
        wandb_model_file = 'model_last'

    nnunet_trainer_kwargs = dict(
        dataset_directory='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020',
        batch_dice=True,
        stage=0,
        unpack_data=True,
        deterministic=False,
        fp16=True,
    )
    if model == 'unet':
        nnunet_trainer = nnUNetTrainerV2(
            '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlansv2.1_plans_2D.pkl',
            0,
            output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2__nnUNetPlansv2.1',
            **nnunet_trainer_kwargs,
        )
        nnunet_trainer.initialize(False)

        net = BraTSnnUNet(nnunet_trainer.network)
        net.pooling = nn.AvgPool2d(3)

        net = load_from_wandb(net, run_id, model_fname=wandb_model_file).to(device)
    elif model == 'resnet':
        nnunet_trainer = nnUNetTrainerV2_ResNetUNet(
            '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_ResNetUNet_v2.1_plans_2D.pkl',
            0,
            output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1',
            **nnunet_trainer_kwargs,
        )
        nnunet_trainer.initialize(False)

        resnet_backbone = nnunet_trainer.network.encoder
        resnet_backbone.default_return_skips = False
        net = nn.Sequential(
            resnet_backbone,
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(1),
            nn.Linear(2048, 1),
        )

    net = load_from_wandb(net, run_id, model_fname=wandb_model_file).to(device)
    net.eval()

    # MAKE PREDICTIONS
    age_hats = list()
    ages = list()
    groups = list()
    for X, a, y in tqdm(dataloader):
        group = group_labels[y.median().item()]

        X = X.repeat((1,4,1,1)).to(device)  # fix input channels

        with torch.no_grad():
            a_hat = h(net(X))

        age_hats.append(a_hat.mean().item())
        ages.append(a.median().item())
        groups.append(group)

    # SAVE PREDICTIONS
    with open(f'/home/jupyter/gama/bruno/data/preds/predictions_{model}_{run_id}_{split}.pkl', 'wb') as f:
        pickle.dump({
            'age_hats': age_hats,
            'ages': ages,
            'groups': groups,
        }, f)
