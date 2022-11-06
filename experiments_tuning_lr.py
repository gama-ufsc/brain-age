from cv2 import transform
import numpy as np
import torch
from torch import nn

from src.trainer import Trainer, ClassificationTrainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models


# if __name__ == '__main__':
#     args = {
#             "seed": 42,
# #             "model_name": 'resnet50',
# #             "model_name": 'efficientnet_b4',
#             "model_name": 'efficientnet_b3',
# #             "model_name": 'inception_resnet_v2',
#             "dropout": 0,
#             "lr": 1e-3,
#             "batch": 64,
#             "epochs": 30,
# #             "scheduler": 'MultiplicativeLR',
#             "scheduler": 'LambdaLR',
#             "normalization": "None",
#             "augmentation": False,
#             "num_workers": 4,
#             "pin_memory": True,
#             "weight_decay": 1e-4,
#             "wandb_entity": 'victorhro',
#             "wandb_project": 'ADNI-testes',
#             "wandb_tags": ['FULL-TRAIN']
#         }
#     name = f"{args['model_name']}-lr_{args['lr']}_b{args['batch']}_scheduler-{args['scheduler']}"

# #     net = models.resnet50(pretrained=True)
# #     net.fc = nn.Sequential(nn.Linear(2048, 1))

# #     net = models.efficientnet_b4(pretrained=True)
# #     net.classifier=nn.Sequential(
# #     nn.Dropout(p=0.4, inplace=True),
# #     nn.Linear(in_features=1792, out_features=1, bias=True))

#     net = models.efficientnet_b3(pretrained=True)
#     net.classifier=nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(in_features=1536, out_features=1, bias=True))

#     Trainer(
#         net,
#         '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=args['epochs'],
#         lr=args['lr'],
#         batch_size=args['batch'],
#         lr_scheduler=args['scheduler'],
#         lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
# #         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
# #         lr_scheduler_params={'start_factor': 1., 'end_factor': 0.1},
#         transforms=torch.Tensor,
#         wandb_project_name=name,
#         wandb_entity=args['wandb_entity'],
#         wandb_project=args['wandb_project'],
#         wandb_tags=args['wandb_tags']
#     ).run(),
    
    
    
# if __name__ == '__main__':
#     for batch in [16, 32, 64]:
#         for lr in [1e-3, 1e-4, 1e-5]:
#             args = {
#                 "seed": 42,
#                 "model_name": 'efficientnet_b4',
#                 "dropout": 0,
#                 "lr": lr,
#                 "batch": batch,
#                 "epochs": 10,
#                 "scheduler": None,
#                 "normalization": "None",
#                 "augmentation": False,
#                 "num_workers": 4,
#                 "pin_memory": True,
#                 "weight_decay": None,
#                 "wandb_entity": 'victorhro',
#                 "wandb_project": 'ADNI-TESTES-LR_BATCH_EfficientNetB4',
#                 "wandb_tags": ['FULL-TRAIN']
#             }
            
#             net = models.efficientnet_b4(pretrained=True)
#             net.classifier  = nn.Sequential(
#             nn.Dropout(p=0.4, inplace=True),
#             nn.Linear(in_features=1792, out_features=1, bias=True))

    
#             name = f"{args['model_name']}-lr_{args['lr']}_b{args['batch']}_norm-{args['normalization']}_scheduler-{args['scheduler']}-with_dense"

#             Trainer(
#                 net,
#                 '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#                 epochs=args['epochs'],
#                 lr=args['lr'],
#                 lr_scheduler=args['scheduler'],
#                 batch_size=args['batch'],
#                 transforms=torch.Tensor,
#                 wandb_project_name=name,
#                 wandb_entity=args['wandb_entity'],
#                 wandb_project=args['wandb_project'],
#                 wandb_tags=args['wandb_tags']
#             ).run()

if __name__ == '__main__':
    for batch in [32, 64]:
        for lr in [1e-3, 1e-4, 1e-5]:
            args = {
                "seed": 42,
                "model_name": 'efficientnet_b4',
                "dropout": 0,
                "lr": lr,
                "batch": batch,
                "epochs": 10,
                "scheduler": None,
                "normalization": "None",
                "augmentation": False,
                "num_workers": 4,
                "pin_memory": True,
                "weight_decay": None,
                "wandb_entity": 'victorhro',
                "wandb_project": 'ADNI-TESTES-LR_BATCH_EfficientNetB4',
                "wandb_tags": ['FULL-TRAIN']
            }
            
            net = models.efficientnet_b4(pretrained=True)
            net.classifier  = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1792, out_features=1, bias=True))

    
            name = f"{args['model_name']}-lr_{args['lr']}_b{args['batch']}_norm-{args['normalization']}_scheduler-{args['scheduler']}-with_dense"

            Trainer(
                net,
                '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
                epochs=args['epochs'],
                lr=args['lr'],
                lr_scheduler=args['scheduler'],
                batch_size=args['batch'],
                transforms=torch.Tensor,
                wandb_project_name=name,
                wandb_entity=args['wandb_entity'],
                wandb_project=args['wandb_project'],
                wandb_tags=args['wandb_tags']
            ).run()