from turtle import forward
import wandb

import torch
from torch import nn


def load_from_wandb(net: nn.Module, run_id: str,
                    project='ADNI-brain-age'):
    best_model_file = wandb.restore(
        'model_best.pth',
        run_path=f"gama/{project}/{run_id}",
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net

class DecoderBraTSnnUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pooling = nn.AvgPool2d(5)

        self.fc = nn.Linear(480,1)

        self.out = nn.Sigmoid()
    
    def forward(self, x):
        # x = self.pooling(x)

        y = self.fc(self.pooling(x).squeeze()).squeeze()

        return self.out(y)

class ClassifierBraTSnnUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pooling = nn.AvgPool2d(5)

        self.fc = nn.Linear(480,50)
    
    def forward(self, x):
        x = self.pooling(x)

        z = self.fc(x.squeeze()).squeeze()

        return z

class BraTSnnUNet(nn.Module):
    def __init__(self, network, freeze=False):
        """`network` must be a nnU-Net `Generic_UNet`.
        """
        super().__init__()

        self.brats_encoder = network.conv_blocks_context

        self.pooling = nn.AvgPool2d(5)

        self.fc = nn.Linear(480,1)

        if freeze:
            self.freeze()

    def forward(self, x):
        for d in range(len(self.brats_encoder)):
            x = self.brats_encoder[d](x)

        x = self.pooling(x)

        y = self.fc(x.squeeze()).squeeze()

        return y

    def freeze(self):
        for param in self.brats_encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.brats_encoder.parameters():
            param.requires_grad = True
