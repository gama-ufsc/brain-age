from cv2 import transform
import numpy as np
import torch
from torch import nn

# trainer v2 = hold out
from src.trainer_v1 import Trainer, ClassificationTrainer
from src.net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models
from nnunet.training.model_restore import restore_model



if __name__ == '__main__':
    args = {
            "seed": 42,
            "model_name": 'unet',
#             "model_name": 'efficientnet_b4',
#             "model_name": 'resnet101',
#             "model_name": 'efficientnet_b3',
#             "model_name": 'inception_resnet_v2',
            "dropout": 0,
            "lr": 1e-3,
            "batch": 64,
            "epochs": 30,
#             "scheduler": 'MultiplicativeLR',
            "scheduler": 'LambdaLR',
            "normalization": "None",
            "augmentation": False,
            "num_workers": 4,
            "pin_memory": True,
            "weight_decay": 1e-4,
            "wandb_entity": 'victorhro',
            "wandb_project": 'ADNI-testes',
            "wandb_tags": ['FULL-TRAIN']
        }
    name = f"dissertacao-{args['model_name']}-lr_{args['lr']}_b{args['batch']}_scheduler-{args['scheduler']}"

#     net = models.resnet50(pretrained=True)
#     net.fc = nn.Sequential(nn.Linear(2048, 1))
#     net = models.resnet101(pretrained=True)
#     net.fc = nn.Sequential(nn.Linear(2048, 1))

#     net = models.efficientnet_b4(pretrained=True)
#     net.classifier=nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(in_features=1792, out_features=1, bias=True))
    
    

#     net = models.efficientnet_b4(pretrained=True)
#     net.classifier=nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(in_features=1536, out_features=1, bias=True))

    net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')
    net.pooling = nn.AvgPool2d(3)

    Trainer(
        net,
        '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
        epochs=args['epochs'],
        lr=args['lr'],
        batch_size=args['batch'],
        lr_scheduler=args['scheduler'],
        lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/args['epochs']},
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#         lr_scheduler_params={'start_factor': 1., 'end_factor': 0.1},
        transforms=torch.Tensor,
        wandb_project_name=name,
        wandb_entity=args['wandb_entity'],
        wandb_project=args['wandb_project'],
        wandb_tags=args['wandb_tags']
    ).run(),
    
    
#     args = {
#             "seed": 42,
# #             "model_name": 'resnet50',
# #             "model_name": 'efficientnet_b4',
#             "model_name": 'inception_resnet_v2',
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

#     import timm
#     net = timm.create_model('inception_resnet_v2', pretrained=True)
#     net.classif = nn.Sequential(nn.Linear(1536, 1))

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

#     net = models.efficientnet_b4(pretrained=True)
#     net.classifier  = nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(in_features=1792, out_features=1, bias=True))
    
#     Trainer(
#         net,
#         '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=args['epochs'],
#         lr=args['lr'],
#         batch_size=args['batch'],
#         lr_scheduler=args['scheduler'],
#         # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/args['epochs'] - 1))},
#         transforms=torch.Tensor,
#         wandb_project_name=name,
#         wandb_entity=args['wandb_entity'],
#         wandb_project=args['wandb_project'],
#         wandb_tags=args['wandb_tags']
#     ).run(),
    

    
#     import timm
#     net = timm.create_model('inception_resnet_v2', pretrained=True)
#     net.classif = nn.Sequential(nn.Linear(1536, 1))
    
#     Trainer(
#         net,
#         '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#         epochs=args['epochs'],
#         lr=args['lr'],
#         batch_size=args['batch'],
#         lr_scheduler=args['scheduler'],
#         # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/args['epochs'] - 1))},
#         transforms=torch.Tensor,
#         wandb_project_name=name,
#         wandb_entity=args['wandb_entity'],
#         wandb_project=args['wandb_project'],
#         wandb_tags=args['wandb_tags']
#     ).run(),
#         net = models.resnet50(pretrained=True)
#         net.fc = nn.Sequential(nn.Linear(2048, 1))
# #         net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
#         Trainer(
#             net,
#             '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#             epochs=args['epochs'],
#             lr=lr,
#             batch_size=args['batch'],
#             lr_scheduler=args['scheduler'],
#             # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/30 - 1))},
#             transforms=torch.Tensor,
#             wandb_project_name=name,
#             wandb_entity=args['wandb_entity'],
#             wandb_project=args['wandb_project'],
#             wandb_tags=args['wandb_tags']
#         ).run()
# net,



#         net = models.efficientnet_b4(pretrained=True)
#         net.classifier  = nn.Sequential(
#             nn.Dropout(p=0.4, inplace=True),
#             nn.Linear(in_features=1792, out_features=1, bias=True)
#         )
        
#         name = f"{args['model_name']}-lr_{args['lr']}_b{args['batch']}_norm-{args['normalization']}_scheduler-{args['scheduler']}-without_dense"

#         Trainer(
#             net,
#             '/home/jupyter/data/ADNI/brats_2mm_preprocessed/ADNI_slices_fix_2mm_split.hdf5',
#             epochs=args['epochs'],
#             lr=lr,
#             batch_size=args['batch'],
#             lr_scheduler=args['scheduler'],
#             # lr_scheduler_params={'lr_lambda': lambda e: 1 -0.9*e/30},
#             lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/args['epochs'] - 1))},
#             transforms=torch.Tensor,
#             wandb_project_name=name,
#             wandb_entity=args['wandb_entity'],
#             wandb_project=args['wandb_project'],
#             wandb_tags=args['wandb_tags']
#         ).run()
# net,

