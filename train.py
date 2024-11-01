import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import F1Score
from torch.utils.data import Subset
from torch.nn import functional as F
from ml_collections import config_dict
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.models import resnet18, mobilenet_v3_small, densenet121


# Add the project name
PROJECT_NAME = 'Trash-recognition'
RAW_DATA_FOLDER = '/content/drive/MyDrive/Proyek/Computer Vision/Trash Recognition/dataset-resized'

# Default Hyperparameter
cfg = config_dict.ConfigDict()
cfg.seed = 1
cfg.img_size = 224
cfg.batch_size = 32
cfg.lr = 0.0001
cfg.arch = 'resnet'
cfg.dropout_rate = 0.5
cfg.fc_neurons = 128

# Load Dataset
def load_data(cfg):
    transforms = Compose([Resize((cfg.img_size, cfg.img_size)), ToTensor()])
    dataset = ImageFolder(RAW_DATA_FOLDER, transform = transforms)
    dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    train_indices, test_indices, y_train, _ = train_test_split(range(len(dataset)), dataset.targets, stratify = dataset.targets, test_size = 0.1, random_state = cfg.seed)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_indices, val_indices, _, _ = train_test_split(train_dataset.indices, y_train, stratify = y_train, test_size = 0.111, random_state = cfg.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return dataset, train_dataset, val_dataset, test_dataset

# Modelling
class Model(pl.LightningModule):

    def __init__(self, cfg, dataset, train_dataset, val_dataset, test_dataset):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        num_classes = len(set(dataset.targets))
        self.f1_score = F1Score(num_classes = num_classes, task = 'multiclass', average = 'macro')
        self.class_weights = torch.Tensor(compute_class_weight(class_weight = 'balanced', classes = np.unique(dataset.targets), y = dataset.targets))


        if cfg.arch == 'resnet':
            self.net = resnet18('DEFAULT')
            self.net.fc = nn.Linear(512, cfg.fc_neurons)
        elif cfg.arch == 'mobilenet':
            self.net = mobilenet_v3_small('DEFAULT')
            self.net.classifier[-1] = nn.Linear(1024, cfg.fc_neurons)
        elif cfg.arch == 'densenet':
            self.net = densenet121('DEFAULT')
            self.net.classifier = nn.Linear(1024, cfg.fc_neurons)
        else: 
            raise ValueError("Architecture should be either 'resnet', 'mobilenet', or 'densenet'.")
        
        self.out = nn.Linear(cfg.fc_neurons, num_classes)

    # Forward Layer
    def forward(self, x):
        x = self.net(x)
        x = F.dropout(x, p = self.cfg.dropout_rate, training = self.training)
        x = F.relu(x)
        x = self.out(x)
        return F.log_softmax(x, dim = 1)
    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y, weight = self.class_weights.to(y_hat.device))
        f1 = self.f1_score(y_hat, y)
        self.log('train_loss', loss, on_step = False, on_epoch = True)
        self.log('train_f1', f1, on_step = False, on_epoch = True)
        return loss
    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y, weight = self.class_weights.to(y_hat.device))
        f1 = self.f1_score(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
    # Prediction Step
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        x, y = batch
        y_hat = self.forward(x)
        return x, y, y_hat
    # Configuration Adam Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, num_workers=2)
    

# Training data
def train(cfg):
    wandb_logger = WandbLogger(job_type = 'train-sweep', config = dict(cfg))
    cfg = wandb.config
    pl.seed_everything(seed = cfg.seed, workers = True)
    dataset, train_dataset, val_dataset, test_dataset = load_data(cfg)
    model = Model(cfg, dataset, train_dataset, val_dataset, test_dataset)
    trainer = pl.Trainer(max_epochs = 10, 
                         accelerator = 'auto',
                         deterministic = True,
                         logger = wandb_logger)
    trainer.fit(model)



if __name__ == '__main__':
    train(cfg)


