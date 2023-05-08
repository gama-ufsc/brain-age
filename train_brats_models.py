import sys

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import nnUNetTrainerV2_ResNetUNet


if __name__ == '__main__':
    args = sys.argv

    if args[-1] == '--all':
        fold = 'all'
        args = args[:-1]
    else:
        fold = 0

    if args[-1].lower() == 'unet':  ### UNet ###
        trainer_kwargs = dict(
            output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2__nnUNetPlansv2.1',
            dataset_directory='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020',
            batch_dice=True,
            stage=0,
            unpack_data=True,
            deterministic=False,
            fp16=True,
        )
        validate_kwargs = dict(
            save_softmax=None,
            validation_folder_name='validation_raw',
            run_postprocessing_on_folds=True,
            overwrite=None,
        )

        trainer = nnUNetTrainerV2(
            '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlansv2.1_plans_2D.pkl',
            fold, **trainer_kwargs
        )

        trainer.initialize(True)

        trainer.run_training()

        trainer.validate(**validate_kwargs)
    elif args[-1].lower() == 'resunet':  ### ResUNet ###
        trainer_kwargs = dict(
            output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1',
            dataset_directory='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020',
            batch_dice=True,
            stage=0,
            unpack_data=True,
            deterministic=False,
            fp16=True,
        )
        validate_kwargs = dict(
            save_softmax=None,
            validation_folder_name='validation_raw',
            run_postprocessing_on_folds=True,
            overwrite=None,
        )

        trainer = nnUNetTrainerV2_ResNetUNet(
            '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_ResNetUNet_v2.1_plans_2D.pkl',
            fold, **trainer_kwargs
        )

        trainer.initialize(True)

        trainer.run_training()

        trainer.validate(**validate_kwargs)
    elif args[-1].lower() == 'pt-resunet':  ### Pre-trained ResUNet ###
        trainer_kwargs = dict(
            output_folder='/home/jupyter/gama/nnUNet/models/nnUNet/2d/Task102_BraTS2020/nnUNetTrainerV2_ResNetUNet__nnUNetPlans_ResNetUNet_v2.1',
            dataset_directory='/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020',
            batch_dice=True,
            stage=0,
            unpack_data=True,
            deterministic=False,
            fp16=True,
            pretrained_resnet=True,
        )
        validate_kwargs = dict(
            save_softmax=None,
            validation_folder_name='validation_raw',
            run_postprocessing_on_folds=True,
            overwrite=None,
        )

        trainer = nnUNetTrainerV2_ResNetUNet(
            '/home/jupyter/gama/nnUNet/data/processed/Task102_BraTS2020/nnUNetPlans_ResNetUNet_v2.1_plans_2D.pkl',
            fold, **trainer_kwargs
        )

        trainer.initialize(True)

        trainer.run_training()

        trainer.validate(**validate_kwargs)
