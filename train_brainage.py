import sys

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import nnUNetTrainerV2_ResNetUNet
from nnunet.training.model_restore import restore_model
import torch
from torchvision import transforms
from torch import nn
import numpy as np

from src.net import load_nnunet_from_wandb
from src.trainer import Trainer
from src.net import BraTSnnUNet

from nnunet.training.model_restore import restore_model

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == '__main__':
    args = sys.argv

    dataset_fpath = './data/interim/ADNI123_slices_fix_2mm_split.hdf5'
    E = 50
    pretrained = None
    run_n = None

    # parse args
    if args[-1] == '--all':
        s = 'train+val'
        args = args[:-1]
        wandb_group = ''
    else:
        s = 'train'
        wandb_group = 'val_'

    try:
        run_n = int(args[-1])
        args = args[:-1]
    except ValueError:
        pass

    if args[-1].lower() in ['old-brats', 'brats', 'imagenet', 'imagenet+brats']:
        pretrained = args[-1].lower()
        args = args[:-1]

    lr = float(args[-1])
    model = args[-2].lower() 

    # fixed kwargs for trainer
    trainer_kwargs = dict(
        epochs=E,
        lr=lr,
        batch_size=64,
#         lr_scheduler='MultiplicativeLR',
#         lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(5*(e/E - 1))},
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ]),
        split=s,
    )
    nnunet_trainer_kwargs = dict(
        dataset_directory='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020',
        batch_dice=True,
        stage=0,
        unpack_data=True,
        deterministic=False,
        fp16=True,
    )

    if model == 'unet':  ### UNet ###
        wandb_group += 'UNet'
        # load nnunet_trainer
        if pretrained == 'brats':
            unet_brats_ids = ['1nfcq3ju', '9ld37s4y', '2r3ii0b4', '2ebpcar2', 'qto04d2p']
            nnunet_trainer = load_nnunet_from_wandb(unet_brats_ids[run_n])
#             nnunet_model_fpath = '/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model.pkl'
#             nnunet_trainer = restore_model(nnunet_model_fpath, checkpoint=nnunet_model_fpath[:-4], train=False)
#             nnunet_trainer.initialize(training=False)

            if s == 'train+val':
                trainer_kwargs['epochs'] = 7

            wandb_group += '+BraTS'
        elif pretrained == 'old-brats':
            net = torch.load('/home/jupyter/gama/bruno/models/brats_model.pt')

            wandb_group += '+old_BraTS'
        else:
            # new trainer, to initialize backbone with random weights
            nnunet_trainer = nnUNetTrainerV2(
                '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlansv2.1_plans_2D.pkl',
                0,
                output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2__nnUNetPlansv2.1',
                **nnunet_trainer_kwargs,
            )
            nnunet_trainer.initialize(False)

            if s == 'train+val':
                trainer_kwargs['epochs'] = 18

        if pretrained != 'old-brats':
            # create brain age model with backbone from nnunet_trainer.network's encoder
            net = BraTSnnUNet(nnunet_trainer.network)
        net.pooling = nn.AvgPool2d(3)

#         if pretrained != 'brats':
#             weight_reset(net)
#             wandb_group += '_reset'

        Trainer(
            net, dataset_fpath,
            wandb_group=wandb_group,
            **trainer_kwargs,
        ).run()
    elif model == 'resnet':  ### ResNet ###
        wandb_group += 'ResNet'
        # load nnunet_trainer
        if pretrained == 'brats':
            resnet_brats_ids = ['6jvz00y4', '1ui6xsgh', '12dollka', '2q3478oi', '3fizs05h']
            nnunet_trainer = load_nnunet_from_wandb(resnet_brats_ids[run_n])
            wandb_group += '+BraTS'
#             nnunet_model_fpath = '/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1/fold_0/model_final_checkpoint.model.pkl'
#             nnunet_trainer = restore_model(nnunet_model_fpath, checkpoint=nnunet_model_fpath[:-4], train=False)
#             nnunet_trainer.initialize(training=False)

#             wandb_group += '+local_BraTS'

            if s == 'train+val':
                trainer_kwargs['epochs'] = 23
        elif pretrained == 'imagenet':
            # new trainer, to initialize ResNet backbone with ImageNet weights
            nnunet_trainer = nnUNetTrainerV2_ResNetUNet(
                '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_ResNetUNet_v2.1_plans_2D.pkl',
                0,
                output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1',
                pretrained_resnet=True,
                **nnunet_trainer_kwargs,
            )
            nnunet_trainer.initialize(False)
            wandb_group += '+ImageNet'
            
            if s == 'train+val':
                trainer_kwargs['epochs'] = 25
        elif pretrained == 'imagenet+brats':
            resnet_imagenet_brats_ids = ['38xp1owi', '16x20wq1', 'e8gc6ih2', '3olfcxsq', '1hsptxr0']
            nnunet_trainer = load_nnunet_from_wandb(resnet_imagenet_brats_ids[run_n])
            wandb_group += '+ImageNet+BraTS'

            if s == 'train+val':
                trainer_kwargs['epochs'] = 38
        else:
            # new trainer, to initialize backbone with random weights
            nnunet_trainer = nnUNetTrainerV2_ResNetUNet(
                '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_ResNetUNet_v2.1_plans_2D.pkl',
                0,
                output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1',
                **nnunet_trainer_kwargs,
            )
            nnunet_trainer.initialize(False)
            
            if s == 'train+val':
                trainer_kwargs['epochs'] = 39

        # create brain age model with backbone from nnunet_trainer.network's encoder
        resnet_backbone = nnunet_trainer.network.encoder
        resnet_backbone.default_return_skips = False
        net = nn.Sequential(
            resnet_backbone,
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(1),
            nn.Linear(2048, 1),
        )

        Trainer(
            net, dataset_fpath,
            wandb_group=wandb_group,
            **trainer_kwargs,
        ).run()
    else:
        print('NOT A VALID MODEL')
