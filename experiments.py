from cv2 import transform
import numpy as np
import torch
from torch import nn

from src.trainer import Trainer, ClassificationTrainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models


if __name__ == '__main__':
    net = torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt')
    net.pooling = nn.AvgPool2d(3)
    Trainer(
        net,
        # '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix_2mm_split.hdf5',
        epochs=30,
        lr=1e-3,
        batch_size=2**6,
        lr_scheduler='MultiplicativeLR',
        # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
        lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ])
    ).run()

    net = models.resnet50(pretrained=True)
    net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
    Trainer(
        net,
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix_2mm_split.hdf5',
        epochs=30,
        lr=1e-3,
        batch_size=2**6,
        lr_scheduler='MultiplicativeLR',
        # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
        lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
        transforms=torch.Tensor,
    ).run()

    net = torch.load('/home/bruno-pacheco/brain-age/models/brainseg_model.pt')
    net.pooling = nn.AvgPool2d(3)
    Trainer(
        net,
        # '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix_2mm_split.hdf5',
        epochs=30,
        lr=1e-3,
        batch_size=2**6,
        lr_scheduler='MultiplicativeLR',
        # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
        lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ])
    ).run()
