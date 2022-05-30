import numpy as np
import torch
import wandb

from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from data import ADNIDatasetForBraTSModel
from net import BraTSnnUNet

from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    ds_train = ADNIDatasetForBraTSModel(
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices.hdf5',
        dataset='train'
    )
    ds_test = ADNIDatasetForBraTSModel(
        '/home/bruno-pacheco/brain-age/data/interim/ADNI_slices.hdf5',
        dataset='test'
    )

    batch_size = 2**9
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
    )

    # load model
    model = torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt')
    model.freeze()  # freezes only the encoder
    model.to(device)

    # setup training
    model.train()
    epochs = 5

    criterion = nn.MSELoss().to(device)
    lr = 1e-2
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # initialize wandb
    wandb.init(
        project="ADNI-brain-age",
        entity="brunompac",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": type(model).__name__,
            "optimizer": opt,
            "lr_scheduler": None,
            "lr_scheduler_params": None,
            "loss_func": criterion,
            "device": device,
        },
    )
    wandb.watch(model)
    print(f"W&B run started ({wandb.run.id})")

    scaler = GradScaler()
    for e in range(epochs):
        e_loss = list()
        e_mae = list()
        for X, y in tqdm(train_loader):
            X, y = X.to(device), y.to(device)

            with autocast():
                y_pred = model(X)
                loss = criterion(y_pred, y.float())

            e_loss.append(loss.item())
            e_mae.append(
                (y_pred - y).abs().mean().item()
            )

            # backprop
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        print(f"Epoch {e+1}")
        print(f"Loss = {np.mean(e_loss)}")
        print(f"MAE = {np.mean(e_mae)}")
