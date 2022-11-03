from cv2 import transform
import numpy as np
import torch
from torch import nn

from src.trainer import Trainer, ClassificationTrainer, SemiSupervisedTrainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models
from nnunet.training.model_restore import restore_model


if __name__ == '__main__':
#     for lr in [1e-2, 1e-4]:
#     net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
#     net.pooling = nn.AvgPool2d(3)
#     net.fc = nn.Sequential(  # classifier output
#         nn.Linear(480,50),
#         nn.Softmax(dim=-1),
#     )
#     ClassificationTrainer(
#         net,
#     '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=200,
#         lr=1e-3,
#         batch_size=64,
# #         lr_scheduler='MultiplicativeLR',
# #         lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/50},
# #         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#         transforms=transforms.Compose([
#             transforms.ToTensor(),
#         ])
#     ).run()


    net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
    net.pooling = nn.AvgPool2d(3)
    net.fc = nn.Linear(480,2)
    SemiSupervisedTrainer(
        net,
        '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
        epochs=100,
        lr=1e-3,
        batch_size=64,
#         lr_scheduler='MultiplicativeLR',
#         lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#                 lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ])
    ).run()


#     for lr in [1e-3, 1e-4, 1e-5]:
#         for batch_size in [16, 32, 64]:
#     net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
#     net.pooling = nn.AvgPool2d(3)
#     Trainer(
#         net,
#         '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=200,
#         lr=1e-3,
#         batch_size=64,
# #         lr_scheduler='MultiplicativeLR',
# #         lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
# #                 lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#         transforms=transforms.Compose([
#             transforms.ToTensor(),
#         ])
#     ).run()


#         nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', checkpoint='/home/jupyter/gama/bruno/models/maper_checkpoint.model', train=False)
#         nnunet_trainer = restore_model('/home/jupyter/gama/bruno/models/maper_checkpoint.model.pkl', train=False)
#         nnunet_trainer.initialize(False)
#         net = BraTSnnUNet(nnunet_trainer.network)
#         net.pooling = nn.AvgPool2d(3)
#         Trainer(
#             net,
#             '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#             epochs=50,
#             lr=1e-3,
#             batch_size=64,
#             lr_scheduler='MultiplicativeLR',
#             lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
# #                 lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#             transforms=transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#         ).run()


#     net = models.resnet50(pretrained=True)
#     net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
#     Trainer(
#         net,
#     '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=30,
#         lr=1e-3,
#         batch_size=2**6,
#         lr_scheduler='MultiplicativeLR',
#         # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#         transforms=torch.Tensor,
#     ).run()


#     net = torch.load('/home/jupyter/gama/bruno/models/brainseg_model.pt')
#     net.pooling = nn.AvgPool2d(3)
#     Trainer(
#         net,
#     '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=30,
#         lr=1e-3,
#         batch_size=2**6,
#         lr_scheduler='MultiplicativeLR',
#         lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
# #         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#         transforms=transforms.Compose([
#             transforms.ToTensor(),
#         ])
#     ).run()
