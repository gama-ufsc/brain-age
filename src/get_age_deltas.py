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

# local imports
from data import ADNIDatasetClassification
from net import BraTSnnUNet, load_from_wandb

PROJ_ROOT = Path('/home/jupyter/gama/bruno')
DATASET_FPATH = PROJ_ROOT/'data/interim/ADNI123_slices_fix_2mm_split_class.hdf5'


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = ADNIDatasetClassification(
        DATASET_FPATH,
        get_age=True,
        dataset='test',
        labels=['CN','MCI','AD'],
    )

    h = lambda x: x*25+75

    meta = pd.read_csv(PROJ_ROOT/'runs_meta.csv')

    dfs = list()
    for i, run_meta in meta.iterrows():
        print(f"{i+1}/{meta.shape[0]}")

        if 'MAPER' in run_meta['Group']:
            nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', checkpoint='/home/jupyter/gama/bruno/models/maper_checkpoint.model', train=False)
            nnunet_trainer.initialize(False)
            net = BraTSnnUNet(nnunet_trainer.network)
            net.pooling = nn.AvgPool2d(3)
        elif 'BrainSeg' in run_meta['Group']:
            net = torch.load('/home/jupyter/gama/bruno/models/brainseg_model.pt')
            net.pooling = nn.AvgPool2d(3)
        elif 'ResNet50' in run_meta['Group']:
            net = models.resnet50(pretrained=False)
            net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
        elif 'BraTS' in run_meta['Group']:
            net = torch.load(PROJ_ROOT/'models/brats_model.pt')
            net.pooling = nn.AvgPool2d(3)
        else:
            nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', train=False)
            nnunet_trainer.initialize(False)
            net = BraTSnnUNet(nnunet_trainer.network)
            net.pooling = nn.AvgPool2d(3)

        net = load_from_wandb(net, run_meta['ID']).to(device)
        net.eval()

        data_loader = DataLoader(data, batch_size=40, shuffle=False)

        age_deltas = list()
        groups = list()
        data_loader = DataLoader(data, batch_size=40, shuffle=False)
        for X, a, y in tqdm(data_loader):
            with torch.no_grad():
                X = X.unsqueeze(1)
                try:
                    n = net.conv1.in_channels
                    X = X.repeat((1,n,1,1))  # fix input channels
                except:
                    pass
                a_pred = h(net(X.to(device))).detach().cpu()
            age_deltas.append(a_pred.numpy().mean() - a.cpu().numpy().mean())
            groups.append(y.cpu().numpy().min())
        age_deltas = np.array(age_deltas)
        groups = np.array(groups)

        df = pd.concat([pd.Series(age_deltas), pd.Series(groups)], axis=1)
        df.columns = ['Delta', 'Group']
        df['Group'] = df['Group'].replace({0:'CN', 1:'MCI', 2:'AD'})

        df['id'] = run_meta['ID']
        df['name'] = run_meta['Name']
        df['run_group'] = run_meta['Group']
        df['model'] = run_meta['model']
        df['split'] = run_meta['split']
        df['val_MAE'] = run_meta['val_MAE']
        df['val_ps_MAE'] = run_meta['val_ps_MAE']

        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv('preds.csv')
