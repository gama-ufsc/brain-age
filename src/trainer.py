import os
import logging
from pathlib import Path
import random

from time import time
from torchvision import transforms
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import wandb
from dotenv import load_dotenv, find_dotenv

from src.data import ADNIDataset, ADNIDatasetClassification
from src.net import BraTSnnUNet


# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load up the entries as environment variables

project_dir = Path(dotenv_path).parent


def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers

class Trainer():
    """Generic trainer for PyTorch NNs.

    Attributes:
        net: the neural network to be trained.
        epochs: number of epochs to train the network.
        lr: learning rate.
        optimizer: optimizer (name of a optimizer inside `torch.optim`).
        loss_func: a valid PyTorch loss function.
        lr_scheduler: if a scheduler is to be used, provide the name of a valid
        `torch.optim.lr_scheduler`.
        lr_scheduler_params: parameters of selected `lr_scheduler`.
        batch_size: batch_size for training.
        device: see `torch.device`.
        wandb_project: W&B project where to log and store model.
        logger: see `logging`.
        random_seed: if not None (default = 42), fixes randomness for Python,
        NumPy as PyTorch (makes trainig reproducible).
    """
    def __init__(self, net: nn.Module, dataset_fpath: Path, epochs=5, lr= 0.01,
                 optimizer: str = 'Adam', loss_func: str = 'MSELoss', h=lambda x: x*25+75,
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 batch_size=16, device=None, transforms=transforms.ToTensor(),
                 wandb_project="ADNI-brain-age", logger=None, split=0,
                 random_seed=42, wandb_group=None) -> None:
        self._is_initalized = False

        self._e = 0  # inital epoch

        self.dataset_fpath = Path(dataset_fpath)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.transforms = transforms

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.net = net.to(self.device)
        self.h = h
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        
        self.split = split

        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.l = logging.getLogger(__name__)
        else:
            self.l = logger

        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.best_val = float('inf')

        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        
        self._Dataset = ADNIDataset

    @classmethod
    def load_trainer(cls, run_id: str, wandb_project="part-counting-regressor",
                     logger=None):
        """Load a previously initialized trainer from wandb.

        Loads checkpoint from wandb and create the instance.

        Args:
            run_id: valid wandb id.
            logger: same as the attribute.
        """
        wandb.init(
            project=wandb_project,
            entity=os.environ['wandb_entity'],
            id=run_id,
            resume='must',
        )

        # load checkpoint file
        checkpoint_file = wandb.restore('checkpoint.tar')
        checkpoint = torch.load(checkpoint_file.name)

        # load model
        # TODO: make it more flexible
        net = torch.load('/home/bruno-pacheco/brain-age/models/brats_model.pt')
        net = net.to(wandb.config['device'])
        net.load_state_dict(checkpoint['model_state_dict'])

        # fix for older versions
        if 'lr_scheduler' not in wandb.config.keys():
            wandb.config['lr_scheduler'] = None
            wandb.config['lr_scheduler_params'] = None

        # create trainer instance
        self = cls(
            epochs=wandb.config['epochs'],
            net=net,
            lr=wandb.config['learning_rate'],
            optimizer=wandb.config['optimizer'],
            loss_func=wandb.config['loss_func'],
            lr_scheduler=wandb.config['lr_scheduler'],
            lr_scheduler_params=wandb.config['lr_scheduler_params'],
            batch_size=wandb.config['batch_size'],
            device=wandb.config['device'],
            logger=logger,
            wandb_project=wandb_project,
            random_seed=wandb.config['random_seed'],
            split=wandb.config['split'],
        )

        if 'best_val' in checkpoint.keys():
            self.best_val = checkpoint['best_val']

        self._e = checkpoint['epoch'] + 1

        self.l.info(f'Resuming training of {wandb.run.name} at epoch {self._e}')

        # load optimizer
        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])

        return self
    
    def _make_loss_func(self):
        self._loss_func = eval(f"nn.{self.loss_func}()")

    def setup_training(self):
        self.l.info('Setting up training')

        Optimizer = eval(f"torch.optim.{self.optimizer}")
        self._optim = Optimizer(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr
        )

        if self.lr_scheduler is not None:
            Scheduler = eval(f"torch.optim.lr_scheduler.{self.lr_scheduler}")
            self._scheduler = Scheduler(self._optim, **self.lr_scheduler_params)

        self._make_loss_func()

        self.l.info('Initializing wandb.')
        self.initialize_wandb()

        self.l.info('Preparing data')
        self.prepare_data()

        self._is_initalized = True

    def initialize_wandb(self):
        wandb.init(
            project=self.wandb_project,
            entity=os.environ['wandb_entity'],
            group=self.wandb_group,
            config={
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "model": type(self.net).__name__,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "lr_scheduler_params": self.lr_scheduler_params,
                "loss_func": self.loss_func,
                "random_seed": self.random_seed,
                "device": self.device,
                "split": self.split,
            },
        )

        wandb.watch(self.net)

        self._id = wandb.run.id

        self.l.info(f"Wandb set up. Run ID: {self._id}")

    def prepare_data(self):
        if self.split == 'train':
            train_data = self._Dataset(
                self.dataset_fpath,
                dataset='train',
                transform=self.transforms,
            )
            val_data = self._Dataset(
                self.dataset_fpath,
                dataset='val',
                transform=self.transforms,
            )
        elif self.split == 'train+val':
            train_data = self._Dataset(
                self.dataset_fpath,
                dataset='train+val',
                transform=self.transforms,
            )
            val_data = self._Dataset(
                self.dataset_fpath,
                dataset='test',
                transform=self.transforms,
            )

        transforms_ = transforms.ToTensor() if 'slices' in self.dataset_fpath.name else torch.Tensor

        # instantiate DataLoaders
        self._dataloader = {
            'train': DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True),
            'val': DataLoader(val_data, batch_size=40, shuffle=False),
        }

    def run(self):
        if not self._is_initalized:
            self.setup_training()

        scaler = GradScaler()
        while self._e < self.epochs:
            self.l.info(f"Epoch {self._e} started ({self._e+1}/{self.epochs})")
            epoch_start_time = time()

            # train
            train_time, train_loss = timeit(self.train_pass)(scaler)

            self.l.info(f"Training pass took {train_time:.3f} seconds")
            self.l.info(f"Training loss = {train_loss}")

            # validation
            val_time, val_scores = timeit(self.validation_pass)()
            self.l.info(f"Validation pass took {val_time:.3f} seconds")

            self.log_val_scores(train_loss, val_scores)

            self.l.info(f"Saving checkpoint")
            self.save_checkpoint()

            if val_scores[-1] < self.best_val:
                self.l.info(f"Saving best model")
                self.save_model(name='model_best')

                self.best_val = val_scores[-1]

            epoch_end_time = time()
            self.l.info(
                f"Epoch {self._e} finished and took "
                f"{epoch_end_time - epoch_start_time:.2f} seconds"
            )

            self._e += 1

        self.l.info(f"Saving model")
        self.save_model(name='model_last')

        wandb.finish()
        self.l.info('Training finished!')

    def log_val_scores(self, train_loss, val_scores):
        val_loss, val_MAE, val_ps_MAE = val_scores
        
        self.l.info(f"Validation {self.loss_func} = {val_loss}")
        self.l.info(f"Validation MAE = {val_MAE}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": val_MAE,
            "val_ps_MAE": val_ps_MAE,
        }, step=self._e, commit=True)

    def train_pass(self, scaler):
        train_loss = 0
        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in tqdm(self._dataloader['train']):
                X = X.to(self.device)
                y = y.to(self.device)

                try:
                    n = self.net.conv1.in_channels
                    X = X.unsqueeze(1).repeat((1,n,1,1))  # fix input channels
                except:
                    pass

                self._optim.zero_grad()

                with autocast():
                    y_hat = self.h(self.net(X))
                    loss = self._loss_func(
                        y_hat.view_as(y),
                        y.float()
                    )

                scaler.scale(loss).backward()

                train_loss += loss.item() * len(y)

                scaler.step(self._optim)
                scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # scale to data size
        train_loss = train_loss / len(self._dataloader['train'].dataset)

        return train_loss

    def validation_pass(self):
        val_loss = 0
        val_MAE = 0
        per_subject_AE = 0
        n_subjects = 0

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self._dataloader['val']:
                X = X.to(self.device)
                y = y.to(self.device)

                try:
                    n = self.net.conv1.in_channels
                    X = X.unsqueeze(1).repeat((1,n,1,1))  # fix input channels
                except:
                    pass

                with autocast():
                    y_hat = self.h(self.net(X))
                    loss_value = self._loss_func(y_hat.view_as(y), y.float()).item()

                val_loss += loss_value * len(y)  # scales to data size

                val_MAE += (y_hat - y).abs().mean().item() * len(y)
                per_subject_AE += (y_hat.median() - y.median()).abs().item()
                n_subjects += 1

        # scale to data size
        len_data = len(self._dataloader['val'].dataset)
        val_loss = val_loss / len_data
        val_MAE = val_MAE / len_data
        val_ps_MAE = per_subject_AE / n_subjects

        return val_loss, val_MAE, val_ps_MAE

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self._e,
            'best_val': self.best_val,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self._optim.state_dict(),
        }

        torch.save(checkpoint, Path(wandb.run.dir)/'checkpoint.tar')
        wandb.save('checkpoint.tar')

    def save_model(self, name='model'):
        fname = f"{name}.pth"
        fpath = Path(wandb.run.dir)/fname

        torch.save(self.net.state_dict(), fpath)
        wandb.save(fname)

        return fpath

class ClassificationTrainer(Trainer):
    def __init__(self, net: nn.Module, dataset_fpath: Path, epochs=5, lr=0.01,
                 optimizer: str = 'Adam', loss_func: str = 'BCEWithLogitsLoss', h=lambda x: x,
                 lr_scheduler: str = None, lr_scheduler_params: dict = None,
                 batch_size=16, device=None, transforms=transforms.ToTensor(),
                 wandb_project="ADNI-AD-CN", logger=None, split=0,
                 random_seed=42, wandb_group=None) -> None:
        super().__init__(net, dataset_fpath, epochs, lr, optimizer, loss_func, h,
                         lr_scheduler, lr_scheduler_params, batch_size, device,
                         transforms, wandb_project, logger, split, random_seed, wandb_group)
    
        self._Dataset = ADNIDatasetClassification

    def train_pass(self, scaler):
        train_loss = 0
        self.net.train()
        with torch.set_grad_enabled(True):
            for X, y in tqdm(self._dataloader['train']):
                X = X.to(self.device)
                y = y.to(self.device).squeeze(-1)

                try:
                    n = self.net.conv1.in_channels
                    X = X.unsqueeze(1).repeat((1,n,1,1))  # fix input channels
                except:
                    pass

                self._optim.zero_grad()

                with autocast():
                    p_hat = self.net(X)  # output in probability mass (logits)

                    loss = self._loss_func(p_hat, y.to(p_hat))

                scaler.scale(loss).backward()

                train_loss += loss.item() * len(y)

                scaler.step(self._optim)
                scaler.update()

            if self.lr_scheduler is not None:
                self._scheduler.step()

        # scale to data size
        train_loss = train_loss / len(self._dataloader['train'].dataset)

        return train_loss

    def validation_pass(self):
        val_loss = 0
        val_acc = 0
        per_subject_acc = 0
        n_subjects = 0

        ys = list()
        y_hats = list()

        self.net.eval()
        with torch.set_grad_enabled(False):
            for X, y in self._dataloader['val']:
                X = X.to(self.device)
                y = y.to(self.device).squeeze(-1)

                with autocast():
                    p_hat = self.net(X)

                    loss_value = self._loss_func(p_hat, y.to(p_hat)).item()

                val_loss += loss_value * len(p_hat)  # scales to data size

                y = y.cpu().detach().numpy()
                ys.append(y)

                y_hat = torch.sigmoid(p_hat).cpu().detach().numpy()
                y_hats.append(y_hat)

                n_subjects += 1

        # scale loss to data size
        len_data = len(self._dataloader['val'].dataset)
        val_loss = val_loss / len_data

        y = np.stack(ys)
        subject_y = y[:,0]

        y_hat = np.stack(y_hats)
        subject_y_hat = y_hat.mean(axis=-1)
        
        val_auc = roc_auc_score(y.flatten(), y_hat.flatten())
        val_ps_auc = roc_auc_score(subject_y, subject_y_hat)

        # use 1/2 as threshold
        y_hat = (y_hat > 0.5).astype(int)
        subject_y_hat = (subject_y_hat > 0.5).astype(int)

        val_acc = (y_hat == y).sum() / len_data
        val_ps_acc = (subject_y_hat == subject_y).sum() / n_subjects

        return val_loss, val_auc, val_ps_auc, val_acc, val_ps_acc

    def log_val_scores(self, train_loss, val_scores):
        val_loss, val_auc, val_ps_auc, val_acc, val_ps_acc = val_scores

        self.l.info(f"Validation {self.loss_func} = {val_loss}")
        self.l.info(f"Validation Accuracy = {100*val_acc:.2f}%")
        self.l.info(f"Validation Accuracy per subject = {100*val_ps_acc:.2f}%")
        self.l.info(f"Validation AUC = {100*val_auc:.2f}%")
        self.l.info(f"Validation AUC per subject = {100*val_ps_auc:.2f}%")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_ps_auc": val_ps_auc,
            "val_acc": val_acc,
            "val_ps_acc": val_ps_acc,
        }, step=self._e, commit=True)

    def _make_loss_func(self):
        if self.loss_func == 'custom':
            def get_distance_cross_entropy(age=lambda i: i, eps=1e-9):
                def distance_cross_entropy(p, y):
                    idx = torch.arange(y.shape[-1]).repeat(y.shape[0],1)
                    w = (age(idx).T - age(y.argmax(-1))).abs().T

                    y_ = 1 - y
                    p_ = 1 - p
                    l = y_ * torch.log(p_ + eps)
                    l = l * w
                    l = -l.sum(-1)

                    return l.mean()  # aggregation

                return distance_cross_entropy

            self._loss_func = get_distance_cross_entropy(age=lambda i: i + 50)
        else:
            super()._make_loss_func()
