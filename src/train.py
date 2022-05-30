from cv2 import transform
import torch
from torch import nn

from trainer import Trainer, ClassificationTrainer
from net import BraTSnnUNet, DecoderBraTSnnUNet, load_from_wandb, ClassifierBraTSnnUNet
from torchvision import transforms, models


if __name__ == '__main__':
    net = models.resnet50(pretrained=True)
    net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
    Trainer(
        net,
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix_2mm.hdf5',
        epochs=50,
        lr=1e-3,
        lr_scheduler='LinearLR',
        lr_scheduler_params={'start_factor': 1., 'end_factor': 0.01},
        batch_size=2**6,
        transforms=torch.Tensor,
    ).run()

    # run_id = 'pq3h4jko'
    # net = torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt')
    # net = load_from_wandb(net, run_id)
    # net.unfreeze()
    # Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
    #     epochs=3,
    #     lr=1e-5,
    #     batch_size=2**8,
    #     transforms=transforms.Compose([
    #         transforms.ToTensor(),
    #     ])
    # ).run()

    # net = torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt')
    # Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
    #     epochs=5,
    #     lr=1e-1,
    #     batch_size=2**9,
    #     transforms=transforms.Compose([
    #         transforms.ToTensor(),
    #     ])
    # ).run()

    # net.pooling = nn.Sequential(
    #     net.pooling,
    #     nn.Dropout(0.4),
    # )
    # trainer = Trainer(
    #     load_from_wandb(net, '1i1abwny'),
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices.hdf5',
    #     epochs=7,
    #     lr=1e-4,
    #     batch_size=2**8,
    #     transforms=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.RandomAffine(
    #             degrees=(0,10),
    #             translate=(.05,.05),
    #         )
    #     ])
    # ).run()
    # net = DecoderBraTSnnUNet()
    # h = lambda x: 50 * x + 50
    # trainer = Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_features_brats.hdf5',
    #     h=h,
    #     epochs=50,
    #     lr=1e-1,
    #     batch_size=2**14,
    #     transforms=torch.Tensor,
    # ).run()

    # net = ClassifierBraTSnnUNet()
    # h = lambda x: 50 * x + 50
    # trainer = ClassificationTrainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_features_brats.hdf5',
    #     h=h,
    #     epochs=50,
    #     lr=1e-2,
    #     batch_size=2**14,
    #     transforms=torch.Tensor,
    # ).run()

    # net = ClassifierBraTSnnUNet()
    # h = lambda x: 50 * x + 50
    # trainer = ClassificationTrainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_features_brats.hdf5',
    #     h=h,
    #     epochs=50,
    #     lr=1e-3,
    #     batch_size=2**14,
    #     transforms=torch.Tensor,
    # ).run()
    # net = models.resnet50(pretrained=False)
    # net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
    # Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
    #     epochs=6,
    #     lr=1e-5,
    #     batch_size=2**8,
    # ).run()

    # net = models.resnet50(pretrained=False)
    # net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
    # Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
    #     epochs=6,
    #     lr=1e-2,
    #     batch_size=2**8,
    # ).run()

    # net = models.resnet50(pretrained=False)
    # net.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))
    # Trainer(
    #     net,
    #     '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices_fix.hdf5',
    #     epochs=6,
    #     lr=1e-4,
    #     batch_size=2**8,
    # ).run()
