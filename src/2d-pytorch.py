import os
import sys
import cv2
import wandb
import gc
import matplotlib.pyplot as plt
import nibabel as nib
import albumentations as A
import numpy as np
import pandas as pd
import seaborn as sns

from time import time

import torch
from torch import nn
from torchvision import models, transforms

from sklearn.metrics import mean_absolute_error
from wandb.keras import WandbCallback

gc.collect()

print("Python", sys.version.split(" ")[0])
print("PyTorch", torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


seed = 20
torch.manual_seed(seed)
np.random.seed(seed)


def read_image(path, is_tensor=False):
    if os.path.basename(path).split(".")[-1] == "nii":

        if is_tensor == True:
            return torch.Tensor(nib.load(path).get_fdata())
        else:
            return nib.load(path).get_fdata()


def read_df(path):
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]
    return df


def crop(img):  # is_tensor=False):
    return img[2:-3, 2:-3, :]
    # return img[5:85, 6:102, :]


def norm_minmax(img):
    # ((img - min) / (max - min)) * 255
    return (((img - img.min()) / (img.max() - img.min())) * 255.0).type(torch.uint8)


def norm_wholebrain(img):# noise=False):
    # torch.Size([c, sag, cor, axi])
    voxel = img[img > 0]
    mean = voxel.mean()
    std = voxel.std()
    
    # torch.Size([c, sag, cor, axi])
    out = img.detach().clone()
    out[out > 0] = (out[out > 0] - mean)/std
    
    

#     if noise == True:
#         out_random = torch.normal(0, 1, size=img.shape)
#         out[img == 0] = out_random[img == 0]
    return out


def load_dataset(path: str, match="none", df_type="3D"):
    """
    load an entire dataset

    Parameters
    ----------
    path: str
        path of dataframe

    match: {'nyul', 'v50', 'clahe', 'none'}, default='none'

    df_type: {'3D', '2D'}
    """
    # loading dataframe
    df = read_df(path)

    if match == "nyul":
        df["filename"] = df["filename"].str.replace("registration", "nyul")

        if df_type == "2D":
            df["filename"] = df["filename"].str.replace(".nii", ".npy")
            df["filename"] = df["filename"].str.replace("slices", "matched_slices_nyul")

    if match == "v1":
        df["filename"] = df["filename"].str.replace("registration", "matched_v1")
        df["filename"] = df["filename"].str.replace(".nii", ".npy")
        if df_type == "2D":
            df["filename"] = df["filename"].str.replace("slices", "matched_slices_v1")

    if match == "clahe":
        df["filename"] = df["filename"].str.replace("registration", "clahed")
        if df_type == "2D":
            df["filename"] = df["filename"].str.replace(".nii", ".npy")
            df["filename"] = df["filename"].str.replace(
                "slices", "matched_slices_clahe"
            )
    if match == "wholebrain":
        if df_type == "2D":
            df["filename"] = df["filename"].str.replace(".nii", ".npy")
            df["filename"] = df["filename"].str.replace("slices", "slices_wholebrain")
            
    if match == "gaussian":
        if df_type == "2D":
            df["filename"] = df["filename"].str.replace(".nii", ".npy")
            df["filename"] = df["filename"].str.replace("slices", "slices_gaussian")

    if match == "wholebrain+noise":
        if df_type == "2D":
            df["filename"] = df["filename"].str.replace(".nii", ".npy")
            df["filename"] = df["filename"].str.replace(
                "slices", "slices_wholebrain+noise"
            )

    return df

def split_datasets(df, split=""):
    """
    Parameters
    ----------
    df: pd.DataFrame

    split: {'none', '2D', '3D'}, default='none'
    """

    if split == "none":
        x = df.filename.to_numpy()
        y = df.age.to_numpy().astype(np.float32)

        return x, y
    else:
        x = df.query("split in @split").filename.to_numpy()
        y = df.query("split in @split").age.to_numpy().astype(np.float32)

        return x, y
    
class MyDataset(torch.utils.data.Dataset):
    """
    Dataset class for read nifti files
    reference: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, files, labels, transform=None):
        """
        Parameters:
        -----------
            files [str]: list of paths

            labels [float]: list of labels

            augmentation [dict]
        """
        self.labels = labels
        self.files = files
        self.transform = transform

        if len(self.files) != len(self.labels) or len(self.files) == 0:
            raise ValueError(
                f"Number of source and target images must be equal and non-zero"
            )

    def __len__(self):
        # Denotes the total number of samples"
        return len(self.files)

    def __getitem__(self, index: int):
        # Generates one sample of data

        # Select sample
        img, label = self.files[index], self.labels[index]

        if os.path.basename(img).split(".")[-1] == "nii":
            img = nib.load(img).get_fdata(dtype=np.float32)
            img = np.stack((img,) * 3)  # convert to rgb - out(channels, h, w)
        else:
            img = np.load(img).astype(np.float32)
            img = np.stack((img,) * 3)  # convert to rgb - out(channels, h, w)

        # transform
        if self.transform:
            """All pre-trained models expect input images normalized in the same way,
            i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
            where H and W are expected to be at least 224.
            The images have to be loaded in to a range of [0, 1]
            and then normalized using mean = [0.485, 0.456, 0.406]
            and std = [0.229, 0.224, 0.225]."""

            # move axis - out(h, w, channels)
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min())
            #             img =  np.expand_dims(img, axis=-1)

            # img = torch.Size([channels, w, h])
            img, label = self.transform(img), torch.Tensor(np.asarray(label))

        return img, label
    
match_ = "gaussian"
split_ = "split_07"

df_3d = load_dataset(
    f"/home/jupyter/brain_age/work_dataframes/csv/{split_}/dataframe3D.csv",
    match=match_,
    df_type="3D",
)

df_2d = load_dataset(
    f"/home/jupyter/brain_age/work_dataframes/csv/{split_}/dataframe2D.csv",
    match=match_,
    df_type="2D",
)


df_2d.drop(df_2d.loc[df_2d["slice"] < 25].index, inplace=True)
df_2d.drop(df_2d.loc[df_2d["slice"] > 64].index, inplace=True)

# N=2
def make_lines(N):
    print("---" * N)


# para split 04

x_train, y_train = split_datasets(df_2d, "train")
x_val_2d, y_val_2d = split_datasets(df_2d, "val")


# OASIS3, ADNI, IXI
x_val_3d, y_val_3d = split_datasets(df_3d, "val")


## testando:
# NT = 400
# NV = 400
# NTEST = 5
# x_train = x_train[:NT]
# y_train = y_train[:NT]

# x_val_2d = x_val_2d[:NV]
# y_val_2d = y_val_2d[:NV]

# x_val_3d = x_val_3d[:NTEST]
# y_val_3d = y_val_3d[:NTEST]

x_train = x_train
y_train = y_train

x_val_2d = x_val_2d
y_val_2d = y_val_2d

x_val_3d = x_val_3d
y_val_3d = y_val_3d

print("=" * 50)
print(f"Quantidade de imagens para treinamento: {len(x_train)}")
print(f"Quantidade de imagens para validação: {len(x_val_2d)}")
print(f"Quantidade de volumes para validação: {len(x_val_3d)}")
print("=" * 50)

print("Match das imagens de treinamento: {}".format(x_train[0].split("/")[-2]))
print("Match das imagens de validação: {}".format(x_val_2d[0].split("/")[-2]))
print("Match dos volumes de validação: {}".format(x_val_3d[0].split("/")[-2]))

print("=" * 50)


# args and dataloader
args = {
    "seed": 20,
    "model_name": 'efficientnetb3',
    "dropout": 0,
    "lr": 1e-4,
    "batch": 64,
    "epochs": 50,
    "normalization": "gaussian",
    "augmentation": False,
    "num_workers":4,
    "pin_memory": True,
    "weight_decay": 1e-4
}


train_params = {
    "batch_size": args['batch'],
    "shuffle": True,
    "drop_last": False,
    "num_workers": args['num_workers'],
    "pin_memory":args['pin_memory']
}
val_params = {
    "batch_size": args['batch'],
    "shuffle": True,
    "drop_last": False,
    "num_workers": args['num_workers'],
    "pin_memory":args['pin_memory']
}
vol_params = {
    "batch_size": 32,
    "shuffle": False,
    "drop_last": False,
    "num_workers": args['num_workers'],
    "pin_memory":args['pin_memory']
}

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

training_set = MyDataset(x_train, y_train, transform=data_transform)
train_loader = torch.utils.data.DataLoader(training_set, **train_params)

val_set = MyDataset(x_val_2d, y_val_2d, transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_set, **val_params)

volume_set = MyDataset(x_val_3d, y_val_3d)
vol_loader = torch.utils.data.DataLoader(volume_set, **vol_params)




# carregando o modelo
# from efficientnet_pytorch import EfficientNet
# def build_model(image_size=(86, 104)):

#     base_model = EfficientNet.from_pretrained(
#         "efficientnet-b3", image_size=image_size, include_top=False, in_channels=3
#     )
#     model = nn.Sequential(
#         base_model, nn.Flatten(), nn.Linear(1536, 1024), nn.ReLU(), nn.Linear(1024, 1)
#     )
#     model.to("cuda")
#     print("")
#     return model

def build_model(image_size=(86, 104), name='efficientnetb3'):
    
    if name == 'efficientnetb3':
        model = models.efficientnet_b3(input_size=image_size,
                                       pretrained=True,
                                       include_top=False)
        
        model.classifier = nn.Sequential(nn.Linear(1536, 1024), nn.ReLU(), nn.Linear(1024, 1))
    
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1))

    model.to(device)
    print("")
    return model



size = (86, 104)
model = build_model(name=args['model_name'])
print('loading:', args['model_name'])

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args['weight_decay'])


print("=" * 50)
print("Training parameters:")
print(args)
print(f"\nloss:\n{criterion}")
print(f"\noptimizer:\n{optimizer}")
print("=" * 50)


project_name = "brain_norm-analysis"
# name = "pytorch_lr-1e-4_batch32_norm-None_epochs-20_dataset-ADNI"
name = f"pytorch-{args['model_name']}-lr_{args['lr']}_batch{args['batch']}_epochs{args['epochs']}_norm-none_dataset-{split_}"
# name='test-warmup'
save_path = f"/home/jupyter/models/PyTorch/norm_analysis/{name}.pth"


from torch.optim.lr_scheduler import LinearLR
linear_decay = LinearLR(optimizer, start_factor=1, total_iters=args['epochs'])

# Mixed Precision
from torch.cuda.amp import GradScaler, autocast


def train_model(model, dataloader, loss_func, optimizer):  # , scheduler=False):
    model.train()
    scaler = GradScaler()  # Creates a GradScaler once at the beginning of training.
    start = time()

    epoch_loss, epoch_mae = [], []
    total_mae = 0.0

    for imgs, labels in dataloader:

        img, label = imgs.to(device), labels.to(
            device
        )  # torch.Size([batch, channels, 86, 104]) , torch.Size([batch])

        label = label.view(label.shape[0], 1)  # torch.Size([32, 1])
        
        # Runs the forward pass with autocasting - Mixed Precision
        # In these regions, CUDA ops run in a dtype chosen by autocast
        # to improve performance while maintaining accuracy.
        with autocast():
            pred = model(img)
            loss = loss_func(pred, label)
            epoch_loss.append(loss.cpu().data)

        
        # Backpropagation - Mixed Precision
        # Gradient scaling helps prevent gradients with small magnitudes
        # from flushing to zero (“underflowing”) when training with mixed precision.
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        for pred_, label_ in zip(pred, label):
            #             pred_ = pred_.view(1,-1)
            mae_ = torch.mean(torch.abs(pred_ - label_))
            epoch_mae.append(mae_.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    epoch_mae = np.asarray(epoch_mae)

    end = time()
    print(
        ("loss:{:.3f}, mae:{:.3f}, Time: {:.2f}s").format(
            epoch_loss.mean(), epoch_mae.mean(), end - start
        )
    )

    results = {"loss": epoch_loss.mean(), "mae": epoch_mae.mean()}
    return results


def validate_model(model, dataloader, loss_func, optimizer):  # , scheduler=False):
    model.eval()
    start = time()

    epoch_loss, epoch_mae = [], []
    total_mae = 0.0

    with torch.no_grad():
        for imgs, labels in dataloader:

            img, label = imgs.to(device), labels.to(
                device
            )  # torch.Size([batch, channels, 86, 104]) , torch.Size([batch])

            label = label.view(label.shape[0], 1)  # torch.Size([32, 1])
            
            # Runs the forward pass with autocasting - Mixed Precision
            with autocast():
                pred = model(img)
                loss = loss_func(pred, label)
                epoch_loss.append(loss.cpu().data)

                for pred_, label_ in zip(pred, label):
                    #             pred_ = pred_.view(1,-1)
                    mae_ = torch.mean(torch.abs(pred_ - label_))
                    epoch_mae.append(mae_.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    epoch_mae = np.asarray(epoch_mae)
    #     print(epoch_mae)

    end = time()
    print(
        ("val_loss:{:.3f}, val_mae:{:.3f}, Time: {:.2f}s").format(
            epoch_loss.mean(), epoch_mae.mean(), end - start
        )
    )

    results = {"loss": epoch_loss.mean(), "mae": epoch_mae.mean()}
    return results


def preprocess_pipeline(img):
    """
    Recebe um volume, transforma em slices e armazena em um dataloader.
    """
    initial_slice = 25
    n_slices = 40
    img = crop(img)  # torch.Size([w, h, d])
    img = norm_wholebrain(img)  # torch.Size([w, h, d])
    img = norm_minmax(img)  # torch.Size([w, h, d])
    img = img.type(torch.float32)  # torch.uint8 to torch.float32
    img = torch.stack(3 * (img,))  # torch.Size([channels, w, h, d])
    # Slicing
    img = img[
        :, :, :, initial_slice : initial_slice + n_slices
    ]  # torch.Size([channels, w, h, N-slices])
    
    img = torch.moveaxis(img, -1, 0)  # torch.Size([N-slices, channels, w, h])
    img = (img - img.min()) / (img.max() - img.min())
    
    # normalize
    data_vol_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # iterate over images
    for idx in range(img.shape[0]):
        img[idx] = data_vol_transform(img[idx])

    return img


@torch.no_grad()
def model_predict(model, X):
    with autocast():
        model.eval()
        ages = model(X)
    return ages.median()


def volume_model(model, imgs, labels, loss_func, optimizer):  # , scheduler=False):
    model.eval()

    start = time()

    epoch_loss, epoch_mae = [], []
    total_mae = 0.0
    
    labels = np.array(labels)
    with torch.no_grad():
        for img_, label_ in zip(imgs, labels):
            
            img = read_image(img_, is_tensor=True)  # torch.Size([w, h, d])
            
            img, label = img.to(device), torch.Tensor(np.array(label_)).to(device)
            
            img = preprocess_pipeline(img)
            
            # Runs the forward pass with autocasting - Mixed Precision
            with autocast():
                pred = model_predict(model, img)

                loss = loss_func(pred, label)
                epoch_loss.append(loss.cpu().data)

            mae_ = torch.mean(torch.abs(pred - label))
            epoch_mae.append(mae_.cpu().data)

    epoch_loss = np.asarray(epoch_loss)
    epoch_mae = np.asarray(epoch_mae)
    
    end = time()
    print(
        ("vol_loss:{:.3f}, vol_mae:{:.3f}, Time: {:.2f}s").format(
            epoch_loss.mean(), epoch_mae.mean(), end - start
        )
    )

    results = {"loss": epoch_loss.mean(), "mae": epoch_mae.mean()}
    return results

def save_checkpoint(filename: str, **checkpoint):
    print(f"Saving checkpoint in {filename}")
    torch.save(checkpoint, filename)
    print('saved')

# FIT
from torch.optim.lr_scheduler import LinearLR
               
linear_decay = LinearLR(optimizer, start_factor=1, total_iters=args["epochs"], end_factor=0)

log_train_mae, log_train_loss = [], []
log_val_mae, log_val_loss = [], []
log_vol_mae, log_vol_loss = [], []

print('START TRAINING')
wandb.init(project=project_name, name=name)

for epoch in range(args["epochs"]):
    print(f'epoch {epoch+1}/{args["epochs"]}')
    train_results = train_model(model, train_loader, nn.MSELoss().to(device), optimizer)
    log_train_mae.append(train_results["mae"])
    log_train_loss.append(train_results["loss"])

    val_results = validate_model(model, val_loader, nn.MSELoss().to(device), optimizer)
    log_val_mae.append(val_results["mae"])
    log_val_loss.append(val_results["loss"])

    vol_results = volume_model(
        model, x_val_3d, y_val_3d, nn.MSELoss().to(device), optimizer
    )
    log_vol_mae.append(vol_results["mae"])
    log_vol_loss.append(vol_results["loss"])
    print("")
    
    wandb.log(
            {
                "loss": train_results["loss"],
                "val_loss": val_results["loss"],
                "mae": train_results["mae"],
                "val_mae": val_results["mae"],
                "val_mae_vol": vol_results["mae"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
    
    checkpoint = {
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": train_results["loss"],
    "val_loss": val_results["loss"],
    "mae": train_results["mae"],
    "val_mae": val_results["mae"],
    "val_mae_vol": vol_results["mae"]}

    # saving checkpoint...
    save_checkpoint(save_path, **checkpoint)
               
    linear_decay.step()
    

wandb.finish()

print("saving model...")
torch.save(model, save_path)
print("saved.")
print("BYE!")