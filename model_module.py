# model_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import kornia.augmentation as K
from config import CONFIG, AugCase

# Inline Kornia augmentation pipeline
class KorniaAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        self.augment = nn.Sequential(
            K.Resize((224,224)), # Fix: Resize all images to 224x224 for torch.stack to merge into one tensor batch
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=30),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1))
        )

    def forward(self, x):
        return self.augment(x)

# Lightning module
class LitResNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        self.case = CONFIG["case"]
        self.augment = KorniaAugmentations() if self.case in [AugCase.CASE1, AugCase.CASE2, AugCase.CASE3] else None

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        # Apply augmentation only in Case 1,2,3
        x, y = batch
        if self.case in [AugCase.CASE1, AugCase.CASE2, AugCase.CASE3] and self.augment:
            x = self.augment(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Apply augmentation only in Case 2,3
        x, y = batch
        if self.case in [AugCase.CASE2, AugCase.CASE3] and self.augment:
            x = self.augment(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Apply augmentation only in Case 3
        x, y = batch
        if self.case == AugCase.CASE3 and self.augment:
            x = self.augment(x)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
