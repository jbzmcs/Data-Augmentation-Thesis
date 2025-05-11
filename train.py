import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from config import CONFIG, AugCase
from data_module import PanelDataModule
from model_module import LitModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with specific augmentation case")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], required=True,
                        help="Choose augmentation case: 1, 2, or 3")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "alexnet", "vgg16", "inception_v3"],
                        help="Select model: resnet18, alexnet, vgg16, or inception_v3")
    return parser.parse_args()


def main():
    args = parse_args()

    # Update the global CONFIG with selected case
    CONFIG["case"] = AugCase(args.case)

    # Prepare Data
    data_module = PanelDataModule(
        dataset_path=CONFIG["dataset_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        model_name=args.model
    )

    # Prepare Model
    num_classes = data_module.num_classes
    model = LitModel(model_name=args.model, num_classes=num_classes, case=CONFIG["case"])

    # Prepare Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.case}/{args.model}",
        filename="{epoch:02d}-{step}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False,
        callbacks=[checkpoint_callback]
    )

    # Train
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()