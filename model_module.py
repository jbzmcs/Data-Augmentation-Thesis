import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics
import kornia.augmentation as K
from config import CONFIG, AugCase


class KorniaAugmentations(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super().__init__()  # Correctly initialize nn.Module
        self.augment = nn.Sequential(
            K.Resize(input_size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=30),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1))
        )

    def forward(self, x):
        return self.augment(x)


class LitModel(pl.LightningModule):
    def __init__(self, model_name="resnet18", num_classes=None, case=AugCase.CASE1):
        super().__init__()
        self.save_hyperparameters()

        if num_classes is None or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

        self.model_name = model_name.lower()
        self.input_size = (299, 299) if "inception" in self.model_name else (224, 224)
        self.case = case
        self.augmenter = KorniaAugmentations(self.input_size)
        self.num_classes = num_classes

        # Init model
        self.model = self._init_model(model_name, num_classes)

        # Metrics
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        # Per-class metrics for test phase
        self.test_precision_per_class = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average=None)
        self.test_recall_per_class = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average=None)
        self.test_f1_per_class = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=None)

        self.metrics = {}  # Populated in on_*_epoch_end

    def _init_model(self, model_name, num_classes):
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == "inception_v3":
            model = models.inception_v3(pretrained=True, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def forward(self, x):
        if self.model_name == "inception_v3" and self.training:
            return self.model(x)  # Returns (output, aux_output) during training
        return self.model(x)

    def augment(self, x):
        return self.augmenter(x)

    def _shared_step(self, batch, stage):
        x, y = batch

        # Apply augmentation based on case and stage
        if self.case == AugCase.CASE1 and stage == "train":
            x = self.augment(x)
        elif self.case == AugCase.CASE2 and stage in ["train", "val"]:
            x = self.augment(x)
        elif self.case == AugCase.CASE3:
            x = self.augment(x)

        logits = self.forward(x)

        # Handle Inception v3 auxiliary loss during training
        if self.model_name == "inception_v3" and stage == "train":
            main_logits, aux_logits = logits
            loss = F.cross_entropy(main_logits, y) + 0.4 * F.cross_entropy(aux_logits, y)
            preds = torch.argmax(main_logits, dim=1)  # Use main output for predictions
        else:
            loss = F.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, "train")
        self.train_acc.update(preds, targets)
        self.train_precision.update(preds, targets)
        self.train_recall.update(preds, targets)
        self.train_f1.update(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_precision", self.train_precision, prog_bar=True)
        self.log("train_recall", self.train_recall, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, "val")
        self.val_acc.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, "test")
        self.test_acc.update(preds, targets)
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_precision_per_class.update(preds, targets)
        self.test_recall_per_class.update(preds, targets)
        self.test_f1_per_class.update(preds, targets)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_precision", self.test_precision, prog_bar=True)
        self.log("test_recall", self.test_recall, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.metrics["train_acc"] = self.train_acc.compute().item()
        self.metrics["train_precision"] = self.train_precision.compute().item()
        self.metrics["train_recall"] = self.train_recall.compute().item()
        self.metrics["train_f1"] = self.train_f1.compute().item()
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.metrics["val_acc"] = self.val_acc.compute().item()
        self.metrics["val_precision"] = self.val_precision.compute().item()
        self.metrics["val_recall"] = self.val_recall.compute().item()
        self.metrics["val_f1"] = self.val_f1.compute().item()
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        self.metrics["test_acc"] = self.test_acc.compute().item()
        self.metrics["test_precision"] = self.test_precision.compute().item()
        self.metrics["test_recall"] = self.test_recall.compute().item()
        self.metrics["test_f1"] = self.test_f1.compute().item()
        # Store per-class metrics
        precision_per_class = self.test_precision_per_class.compute().tolist()
        recall_per_class = self.test_recall_per_class.compute().tolist()
        f1_per_class = self.test_f1_per_class.compute().tolist()
        for i in range(self.num_classes):
            self.metrics[f"test_precision_class_{i}"] = precision_per_class[i]
            self.metrics[f"test_recall_class_{i}"] = recall_per_class[i]
            self.metrics[f"test_f1_class_{i}"] = f1_per_class[i]
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()
        self.test_f1_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }