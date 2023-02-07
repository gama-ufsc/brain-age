from cv2 import transform
import numpy as np
import torch
from torch import nn

from src.trainer import Trainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models
from nnunet.training.model_restore import restore_model


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':
    dataset_fpath = './data/interim/ADNI123_slices_fix_2mm_split.hdf5'
    E = 30
    n_runs = 5
    s = 'train+val'

    for _ in range(n_runs):
        nnunet_trainer = restore_model('/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1/all/model_final_checkpoint.model.pkl', train=False)
        nnunet_trainer.initialize(False)
        resnet_encoder = nnunet_trainer.network.encoder
        resnet_encoder.default_return_skips = False
        net = nn.Sequential(
            resnet_encoder,
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        Trainer(
            net,
            dataset_fpath,
            epochs=E,
            lr=1e-3,
            batch_size=64,
            lr_scheduler='MultiplicativeLR',
            lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
            transforms=transforms.Compose([
                transforms.ToTensor(),
            ]),
            split=s,
            wandb_group='ADNI23_LR_ResNet50+ImageNet+BraTS',
        ).run()

    #     net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
#         nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', train=False)
#         nnunet_trainer.initialize(False)
#         net = BraTSnnUNet(nnunet_trainer.network)
#         net.pooling = nn.AvgPool2d(3)
#         net.apply(weight_reset)
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=transforms.Compose([
#                 transforms.ToTensor(),
#             ]),
#             split=s,
#             wandb_group='ADNI23_LR_UNet'
#         ).run()

#         net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
#         net.pooling = nn.AvgPool2d(3)
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=transforms.Compose([
#                 transforms.ToTensor(),
#             ]),
#             split=s,
#             wandb_group='train_ADNI23_LR_UNet+BraTS'
#         ).run()

#         nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', checkpoint='/home/jupyter/gama/bruno/models/maper_checkpoint.model', train=False)
#         nnunet_trainer.initialize(False)
#         net = BraTSnnUNet(nnunet_trainer.network)
#         net.pooling = nn.AvgPool2d(3)
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=transforms.Compose([
#                 transforms.ToTensor(),
#             ]),
#             split=s,
#             wandb_group='ADNI23_LR_UNet+old_MAPER'
#         ).run()

#     #     net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
#         net = torch.load('/home/jupyter/gama/bruno/models/brainseg_model.pt')
#         net.pooling = nn.AvgPool2d(3)
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=transforms.Compose([
#                 transforms.ToTensor(),
#             ]),
#             split=s,
#             wandb_group='ADNI23_LR_UNet+old_FSL_BrainSeg'
#         ).run()

#         net = models.resnet50(pretrained=False)
#         net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=torch.Tensor,
#             split=s,
#             wandb_group='ADNI23_LR_ResNet50'
#         ).run()

#         net = models.resnet50(pretrained=True)
#         net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
#         Trainer(
#             net,
#             dataset_fpath,
#             epochs=E,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#             transforms=torch.Tensor,
#             split=s,
#             wandb_group='ADNI23_LR_ResNet50+ImageNet'
#         ).run()
