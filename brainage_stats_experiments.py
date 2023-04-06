from cv2 import transform
import numpy as np
import torch
from torch import nn

from src.trainer import StatisticalAnalysisTrainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models
from nnunet.training.model_restore import restore_model


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

if __name__ == '__main__':
    dataset_fpath = './data/interim/ADNI123_slices_fix_2mm_split.hdf5'
    E = 30
    s = 'train+val'

    nnunet_model_fpath = '/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1/all_pretrain/model_final_checkpoint.model.pkl'
    nnunet_trainer = restore_model(nnunet_model_fpath, checkpoint=nnunet_model_fpath[:-4], train=False)
    nnunet_trainer.initialize(training=False)
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
    StatisticalAnalysisTrainer(
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
#     net.pooling = nn.AvgPool2d(3)
#     StatisticalAnalysisTrainer(
#         net,
#         dataset_fpath,
#         epochs=E,
#         lr=1e-3,
#         batch_size=64,
#         lr_scheduler='MultiplicativeLR',
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
#         transforms=transforms.Compose([
#             transforms.ToTensor(),
#         ]),
#         split=s,
#         wandb_group='train_ADNI23_LR_UNet+BraTS'
#     ).run()
