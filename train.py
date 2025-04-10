import argparse
import torch
import pytorch_lightning as pl
from config import CONFIG, AugCase
from data_module import PanelDataModule
from model_module import LitResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with specific augmentation case")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], required=True,
                        help="Choose augmentation case: 1, 2, or 3")
    return parser.parse_args()


def main():
    args = parse_args()

    # Update the global CONFIG with selected case
    CONFIG["case"] = AugCase(args.case)

    # Prepare Data
    data_module = PanelDataModule(
        dataset_path=CONFIG["dataset_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )

    # Prepare Model
    num_classes = data_module.num_classes
    model = LitResNet(num_classes=num_classes)

    # Prepare Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=False
    )

    # Train
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
